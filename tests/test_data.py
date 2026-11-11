import json
from pathlib import Path

import pytest
from PIL import Image

from src.data import audit_dataset, fingerprint, index_split


@pytest.fixture
def roots(tmp_path):
    roots = {level: str(tmp_path / level) for level in ("line", "paragraph")}
    for i, level in enumerate(roots):
        for j, split in enumerate(("train", "test")):
            group = Path(roots[level]) / f"{split}_data" / ("1" if split == "train" else "250")
            group.mkdir(parents=True)
            Image.new("RGB", (12 + i, 10), (20 + j * 100, i * 80, j * 100)).save(group / "1.jpg")
            (group / "label.json").write_text(json.dumps({"1.jpg": "Tiếng Việt có dấu"}, ensure_ascii=False), encoding="utf-8")
    return roots


def test_full_pool_and_unicode(roots):
    train = index_split(roots, "train")
    assert {s.level for s in train} == {"line", "paragraph"}
    assert len(train) == 2 and train[0].text == "Tiếng Việt có dấu"
    before = fingerprint(train)
    Path(train[0].image_path).write_bytes(Path(train[1].image_path).read_bytes())
    assert fingerprint(train) != before


def test_missing_image_and_unlabeled_image(roots):
    sample = index_split(roots, "train")[0]
    Path(sample.image_path).unlink()
    with pytest.raises(ValueError, match="Image/label mismatch"):
        index_split(roots, "train")


def test_pixel_leakage_and_text_candidates(roots):
    assert audit_dataset(roots)["integrity_passed"]
    train = index_split(roots, "train")[0]
    test = index_split(roots, "test")[1]
    Path(test.image_path).write_bytes(Path(train.image_path).read_bytes())
    report = audit_dataset(roots)
    assert not report["integrity_passed"]
    assert report["cross_split_pixel_duplicates"]
    assert report["cross_split_exact_text"]


def test_duplicate_json_keys(roots):
    label = Path(roots["line"]) / "train_data/1/label.json"
    label.write_text('{"1.jpg":"a", "1.jpg":"b"}')
    with pytest.raises(ValueError, match="Duplicate"):
        index_split(roots, "train")


def test_unlabeled_images_block_full_protocol(roots):
    extra = Path(roots["line"]) / "train_data/1/extra.jpg"
    Image.new("RGB", (10, 10), "red").save(extra)
    with pytest.raises(ValueError, match="unlabeled"):
        index_split(roots, "train")
    report = audit_dataset(roots)
    assert not report["integrity_passed"]
    assert report["split_integrity_passed"]
    assert report["unlabeled_images"] == [str(extra)]
    allowed = audit_dataset(roots, unlabeled_policy="exclude")
    assert allowed["integrity_passed"] and allowed["excluded_images"] == [str(extra)]
