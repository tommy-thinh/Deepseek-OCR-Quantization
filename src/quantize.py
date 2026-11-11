"""Merge E1 in BF16, verify predictions, THEN serialize bitsandbytes PTQ."""
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from .common import digest, file_hash, gpu_device, load_config, new_output, parser, seed_all, write_json
from .data import index_split, require_audit
from .evaluate import predict
from .merge import verified_merge
from .metrics import grouped_quality
from .model import load_base, precision_report, validate_adapter
from .preprocessing import OCRCollator


def main():
    p = parser("E2a/E2b PTQ from a validated E1 adapter, never from E3")
    p.add_argument("--merge-check-samples", type=int, help="Default: complete test set for full PTQ, two samples for smoke")
    p.add_argument("--smoke", action="store_true", help="Test conversion with a smoke E1 adapter; outputs cannot be used as full experiments")
    p.add_argument("--source-adapter", help="Override E1 source for smoke verification only")
    p.add_argument("--output", help="New quantized artifact directory")
    p.add_argument("--merged-model", help="New or previously verified merged BF16 directory")
    p.add_argument("--require-identical-predictions", action="store_true", help="Optional strict gate; BF16 rounding can change greedy continuations")
    args = p.parse_args()
    cfg = load_config(args.config)
    if cfg["experiment"] not in {"E2a", "E2b"}:
        p.error("PTQ supports E2a and E2b only")
    if args.source_adapter and not args.smoke:
        p.error("--source-adapter is only supported with --smoke; edit the full experiment YAML otherwise")
    if args.smoke:
        cfg["source_adapter"] = args.source_adapter or "models/smoke-E1/final"
        cfg["merged_model"] = "models/smoke-E1-merged"
        cfg["artifact"] = f"models/smoke-{cfg['experiment']}/quantized"
    if args.output:
        cfg["artifact"] = args.output
    if args.merged_model:
        cfg["merged_model"] = args.merged_model
    if args.merge_check_samples is not None and args.merge_check_samples < 2:
        p.error("Merge verification requires at least two samples, covering both levels")
    device = gpu_device()
    seed_all(cfg["seed"])
    audit = require_audit(cfg)
    all_samples = index_split(cfg["data"]["roots"], "test", allow_unlabeled=cfg["data"]["unlabeled_policy"] == "exclude")
    control_samples = min(args.merge_check_samples or (2 if args.smoke else len(all_samples)), len(all_samples))
    metadata = validate_adapter(cfg["source_adapter"], cfg, "E1", allow_smoke=args.smoke)
    if metadata["train_fingerprint"] != audit["train_fingerprint"] or metadata["test_fingerprint"] != audit["test_fingerprint"]:
        raise ValueError("E1 adapter trained with a different dataset")
    artifact = new_output(cfg["artifact"])
    merged_dir = Path(cfg["merged_model"])
    provenance_key = digest({"metadata": metadata, "adapter_sha256": file_hash(Path(cfg["source_adapter"]) / "adapter_model.safetensors")})
    if merged_dir.exists():
        check = json.loads((merged_dir / "merge_verification.json").read_text())
        if not check["passed"] or check["source_protocol_hash"] != provenance_key or check["generation"] != cfg["generation"]:
            raise ValueError("Existing merged model has a different E1 source or protocol")
        if args.require_identical_predictions and not check["greedy_tokens_identical"]:
            raise RuntimeError("Existing merged BF16 control fails the requested exact-greedy gate")
        if len(check["sample_ids"]) < control_samples:
            raise ValueError("Existing merge control has fewer samples than requested; choose a new --merged-model directory")
    else:
        base, tokenizer = load_base(cfg, device, precision="bf16")
        collator = OCRCollator.from_model(base, tokenizer, cfg["preprocessing"])
        model = PeftModel.from_pretrained(base, cfg["source_adapter"], is_trainable=False).eval()
        by_level = [[s for s in all_samples if s.level == level] for level in ("line", "paragraph")]
        samples = [group[i] for i in range(control_samples) for group in by_level if i < len(group)][:control_samples]
        before = [predict(model, tokenizer, collator, s, device, cfg["generation"])[2] for s in samples]
        # https://huggingface.co/docs/peft/developer_guides/lora#merge-lora-weights-into-the-base-model
        merged, weight_audit = verified_merge(model)
        merged = merged.to(dtype=torch.bfloat16).eval()
        after = [predict(merged, tokenizer, collator, s, device, cfg["generation"])[2] for s in samples]
        def records(tokens):
            return [{"level": s.level, "reference": s.text, "prediction": tokenizer.decode(ids, skip_special_tokens=True)} for s, ids in zip(samples, tokens)]
        check = {"passed": True, "weight_audit": weight_audit, "greedy_tokens_identical": before == after, "source_protocol_hash": provenance_key,
                 "control_scope": "full_test" if len(samples) == len(all_samples) else "subset",
                 "generation": cfg["generation"], "sample_ids": [s.id for s in samples],
                 "unmerged_token_ids": before, "merged_token_ids": after,
                 "unmerged_quality": grouped_quality(records(before)), "merged_bf16_quality": grouped_quality(records(after)),
                 "criterion": "Exact per-layer merge arithmetic and finite weights; greedy output differences are reported separately",
                 "rounding_note": "Merged BF16 and BF16 base + separate LoRA matmuls need not have identical greedy decoding. Use the saved merged BF16 control when attributing PTQ quality changes."}
        if args.require_identical_predictions and not check["greedy_tokens_identical"]:
            write_json(artifact / "merge_verification_failed.json", check)
            raise RuntimeError("BF16 merged predictions differ from E1; inspect rounding changes before PTQ")
        merged_dir.mkdir(parents=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        write_json(merged_dir / "merge_verification.json", check)
        del model, merged, base
        gc.collect()
        torch.cuda.empty_cache()
    quantized, tokenizer = load_base(cfg, device, source=str(merged_dir))
    report = precision_report(quantized)
    if not report["quantized_modules"]:
        raise RuntimeError("Requested PTQ produced no quantized modules")
    quantized.save_pretrained(artifact, safe_serialization=True)
    tokenizer.save_pretrained(artifact)
    write_json(artifact / "precision.json", report)
    manifest = {"experiment": cfg["experiment"], "model": cfg["model"], "source_adapter": cfg["source_adapter"],
                "smoke": args.smoke, "train_fingerprint": audit["train_fingerprint"], "test_fingerprint": audit["test_fingerprint"],
                "source_protocol_hash": provenance_key, "precision": cfg["precision"], "merge_verification": check,
                "skipped_modules": cfg["model"]["skip_modules"], "quantization": quantized.config.quantization_config.to_dict()}
    del quantized
    gc.collect()
    torch.cuda.empty_cache()
    # Verify serialized quantization state actually reloads and executes both image granularities.
    reloaded, tokenizer = load_base(cfg, device, source=str(artifact))
    reloaded.eval()
    collator = OCRCollator.from_model(reloaded, tokenizer, cfg["preprocessing"])
    samples = index_split(cfg["data"]["roots"], "test", allow_unlabeled=cfg["data"]["unlabeled_policy"] == "exclude")
    manifest["reload_predictions"] = [{"id": s.id, "prediction": predict(reloaded, tokenizer, collator, s, device, cfg["generation"])[0]}
                                      for s in [next(s for s in samples if s.level == level) for level in ("line", "paragraph")]]
    write_json(artifact / "quantization_provenance.json", manifest)


if __name__ == "__main__":
    main()
