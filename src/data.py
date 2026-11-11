"""Strict indexing plus cross-split leakage checks for the pooled UIT-HWDB data."""
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .common import digest, load_config, parser, write_json


@dataclass(frozen=True)
class Sample:
    id: str
    level: str
    split: str
    group: str
    image_path: str
    text: str


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate label key: {key}")
        result[key] = value
    return result


def index_split(roots, split, allow_unlabeled=False):
    if split not in {"train", "test"}:
        raise ValueError(split)
    samples = []
    for level, root in sorted(roots.items()):
        parent = Path(root) / f"{split}_data"
        if not parent.is_dir():
            raise FileNotFoundError(parent)
        groups = sorted((p for p in parent.iterdir() if p.is_dir()), key=lambda p: int(p.name) if p.name.isdigit() else -1)
        if not groups or any(not p.name.isdigit() for p in groups):
            raise ValueError(f"Expected numeric sample groups in {parent}")
        for group in groups:
            labels = json.loads((group / "label.json").read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
            if not isinstance(labels, dict) or not labels:
                raise ValueError(f"Empty/invalid labels: {group}")
            actual = {p.name for p in group.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
            if set(labels) - actual or (actual - set(labels) and not allow_unlabeled):
                raise ValueError(f"Image/label mismatch in {group}: missing={set(labels)-actual}, unlabeled={actual-set(labels)}")
            for name, text in sorted(labels.items()):
                if Path(name).name != name or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Invalid filename/transcription: {group}/{name}")
                samples.append(Sample(f"{level}/{split}/{group.name}/{name}", level, split, group.name, str(group / name), text))
    return samples


def fingerprint(samples):
    # Include image bytes, not absolute paths, so manifests survive relocation.
    return digest([{**{k: v for k, v in asdict(s).items() if k != "image_path"}, "image_sha256": hashlib.sha256(Path(s.image_path).read_bytes()).hexdigest()} for s in samples])


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def audit_dataset(roots, unlabeled_policy="error"):
    if unlabeled_policy not in {"error", "exclude"}:
        raise ValueError("unlabeled_policy must be error or exclude")
    train, test = index_split(roots, "train", allow_unlabeled=True), index_split(roots, "test", allow_unlabeled=True)
    unlabeled = []
    for level, root in sorted(roots.items()):
        for split in ("train", "test"):
            for group in (Path(root) / f"{split}_data").iterdir():
                labels = json.loads((group / "label.json").read_text(encoding="utf-8"))
                unlabeled.extend(str(image) for image in group.iterdir() if image.suffix.lower() in {".jpg", ".jpeg", ".png"} and image.name not in labels)
    by_pixel, by_text = defaultdict(list), defaultdict(list)
    for sample in train + test:
        with Image.open(sample.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixel_key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        by_pixel[pixel_key].append(sample)
        by_text[normalized_text(sample.text)].append(sample)

    def cross_split(groups):
        return [[s.id for s in group] for group in groups.values() if len({s.split for s in group}) > 1]

    group_overlap = sorted({s.group for s in train} & {s.group for s in test})
    # Long line text contained in test paragraphs (and the reverse) is a leakage candidate,
    # not proof: handwritten copies of the same source text can be independent images.
    containment = []
    paragraphs = {split: [(s, normalized_text(s.text)) for s in (train if split == "train" else test) if s.level == "paragraph"] for split in ("train", "test")}
    for line in [s for s in train + test if s.level == "line" and len(normalized_text(s.text)) >= 40]:
        text = normalized_text(line.text)
        for paragraph, paragraph_text in paragraphs["test" if line.split == "train" else "train"]:
            if text in paragraph_text:
                containment.append([line.id, paragraph.id])
    pixel_overlap = cross_split(by_pixel)
    return {"counts": dict(Counter(f"{s.level}/{s.split}" for s in train + test)),
            "train_fingerprint": fingerprint(train), "test_fingerprint": fingerprint(test),
            "cross_split_pixel_duplicates": pixel_overlap,
            "cross_split_exact_text": cross_split(by_text), "cross_split_group_ids": group_overlap,
            "cross_level_text_containment": containment,
            "same_split_cross_level_pixel_duplicates": [[s.id for s in group] for group in by_pixel.values() if len({s.level for s in group}) > 1 and len({s.split for s in group}) == 1],
            "unlabeled_images": sorted(unlabeled), "unlabeled_policy": unlabeled_policy,
            "excluded_images": sorted(unlabeled) if unlabeled_policy == "exclude" else [],
            "split_integrity_passed": not pixel_overlap and not group_overlap,
            "integrity_passed": not pixel_overlap and not group_overlap and (not unlabeled or unlabeled_policy == "exclude"),
            "interpretation": "Text matches/containment require manual provenance review; group IDs are treated conservatively as shared source IDs. Same-split cross-level views are retained."}


def require_audit(cfg):
    path = Path(cfg["data"]["audit"])
    if not path.exists():
        raise RuntimeError("Run python -m src.data --config configs/E0.yaml before experiments")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["integrity_passed"]:
        raise RuntimeError("Dataset audit found cross-split image/source overlap; resolve it before experiments")
    if report["unlabeled_policy"] != cfg["data"]["unlabeled_policy"]:
        raise RuntimeError("Stale unlabeled-image policy in audit")
    for split in ("train", "test"):
        if fingerprint(index_split(cfg["data"]["roots"], split, allow_unlabeled=cfg["data"]["unlabeled_policy"] == "exclude")) != report[f"{split}_fingerprint"]:
            raise RuntimeError(f"Stale {split} data audit")
    return report


def main():
    p = parser("Validate complete pooled line/paragraph datasets and investigate leakage")
    args = p.parse_args()
    cfg = load_config(args.config)
    report = audit_dataset(cfg["data"]["roots"], cfg["data"]["unlabeled_policy"])
    write_json(cfg["data"]["audit"], report)
    print(json.dumps({k: report[k] for k in ("counts", "integrity_passed")}, indent=2))
    if not report["integrity_passed"]:
        raise SystemExit("Incomplete labels or cross-split leakage detected; see audit report")


if __name__ == "__main__":
    main()
