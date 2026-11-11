"""Compare real OCR batches with pinned upstream infer(), without model weights/generation."""
import contextlib
import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common import load_config, write_json
from src.data import index_split
from src.preprocessing import OCRCollator
from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/E0.yaml")
    parser.add_argument("--output", default="tmp/preprocessing_parity.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    options = {"revision": cfg["model"]["revision"], "trust_remote_code": True, "local_files_only": True}
    config = AutoConfig.from_pretrained(cfg["model"]["name"], **options)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    helpers = importlib.import_module(type(config).__module__)
    collator = OCRCollator(tokenizer, helpers, cfg["preprocessing"])
    samples = {s.id: s for s in index_split(cfg["data"]["roots"], "test")}
    ids = {"line/test/250/1.jpg"}
    for file in Path("results/standard").glob("*/predictions.jsonl"):
        ids.update(r["id"] for r in map(json.loads, file.read_text(encoding="utf-8").splitlines()) if r["at_generation_limit"])
    report = {"model_revision": cfg["model"]["revision"], "scope": "CPU preprocessing capture; no model weights, forward pass, or real generation", "samples": []}
    for sample_id in sorted(ids):
        sample = samples[sample_id]
        captured = {}

        def generate(input_ids, **kwargs):
            captured.update(input_ids=input_ids, **kwargs)
            return torch.cat([input_ids, torch.tensor([[tokenizer.eos_token_id]])], dim=1)

        dummy = SimpleNamespace(disable_torch_init=lambda: None, generate=generate)
        with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor), patch.object(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()):
            helpers.DeepseekOCRForCausalLM.infer(dummy, tokenizer, prompt=cfg["preprocessing"]["prompt"], image_file=sample.image_path,
                                              output_path="tmp/preprocessing-parity", base_size=cfg["preprocessing"]["base_size"], image_size=cfg["preprocessing"]["image_size"], crop_mode=cfg["preprocessing"]["crop_mode"], eval_mode=True)
        batch = collator([sample], training=False)
        for key in ("input_ids", "images_seq_mask", "images_spatial_crop"):
            assert torch.equal(batch[key], captured[key]), (sample_id, key)
        for actual, expected in zip(batch["images"][0], captured["images"][0]):
            assert actual.dtype == expected.dtype and torch.equal(actual, expected), (sample_id, "image tensor")
        assert batch["attention_mask"].eq(1).all()
        training = collator([sample], training=True)
        prefix = batch["input_ids"].shape[1]
        assert torch.equal(training["input_ids"][:, :prefix], batch["input_ids"])
        assert training["labels"][:, :prefix].eq(-100).all()
        assert training["labels"][0, prefix:].tolist() == tokenizer.encode(sample.text, add_special_tokens=False) + [tokenizer.eos_token_id]
        report["samples"].append({"id": sample_id, "passed": True, "image_tokens": int(batch["images_seq_mask"].sum()),
                                  "crop_grid": batch["images_spatial_crop"].tolist()[0], "prompt_tokens": prefix})
        report["upstream_generation"] = {k: captured[k] for k in ("temperature", "max_new_tokens", "no_repeat_ngram_size", "use_cache")}
    report["benchmark_generation"] = cfg["generation"]
    report["all_passed"] = True
    write_json(args.output, report)
    print(f"Upstream preprocessing parity passed on {len(ids)} real test images; no model generation ran.")


if __name__ == "__main__":
    main()
