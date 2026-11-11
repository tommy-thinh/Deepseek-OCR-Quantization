"""Quality evaluation and synchronized end-to-end inference benchmarking."""
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from .common import digest, environment, gpu_device, load_config, new_output, parser, seed_all, tree_bytes, write_json
from .data import index_split, require_audit
from .metrics import error_analysis, grouped_quality
from .model import load_experiment, precision_report, validate_adapter
from .preprocessing import OCRCollator, to_device


def predict(model, tokenizer, collator, sample, device, generation):
    batch = to_device(collator([sample], training=False), device)
    prefix_length = batch["input_ids"].shape[1]
    if prefix_length + generation["max_new_tokens"] > collator.cfg["max_length"]:
        raise ValueError(f"{sample.id}: prompt + max_new_tokens exceed context limit")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(**batch, **generation, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = outputs[0, prefix_length:]
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated), generated.tolist()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(position)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)


def evaluate(cfg, output, limit=None, benchmark=True):
    output = new_output(output)
    device = gpu_device()
    seed_all(cfg["seed"])
    audit = require_audit(cfg)
    if cfg["experiment"] in {"E1", "E3"}:
        provenance = validate_adapter(cfg["adapter"], cfg, cfg["experiment"], allow_smoke=limit is not None)
        if any(provenance[key] != audit[key] for key in ("train_fingerprint", "test_fingerprint")):
            raise ValueError("Adapter and evaluation dataset fingerprints differ")
    if cfg["experiment"] in {"E2a", "E2b"}:
        provenance = json.loads((Path(cfg["artifact"]) / "quantization_provenance.json").read_text())
        if any(provenance[key] != audit[key] for key in ("train_fingerprint", "test_fingerprint")):
            raise ValueError("Quantized artifact and evaluation dataset fingerprints differ")
    samples = index_split(cfg["data"]["roots"], "test", allow_unlabeled=cfg["data"]["unlabeled_policy"] == "exclude")
    # Round-robin levels ensures even a 2-sample smoke exercises line and paragraph.
    levels = sorted({s.level for s in samples})
    queues = [[s for s in samples if s.level == level] for level in levels]
    samples = [queue[i] for i in range(max(map(len, queues))) for queue in queues if i < len(queue)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    model, tokenizer = load_experiment(cfg, device, allow_smoke=limit is not None)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - start
    load_peak = torch.cuda.max_memory_allocated(device)
    original = model.get_base_model() if hasattr(model, "peft_config") else model
    collator = OCRCollator.from_model(original, tokenizer, cfg["preprocessing"])
    write_json(output / "environment.json", environment())
    write_json(output / "config.json", cfg)
    write_json(output / "precision.json", precision_report(model))
    warmup = cfg["benchmark"]["warmup"] if benchmark else 0
    for i in range(warmup):
        predict(model, tokenizer, collator, samples[i % len(samples)], device, cfg["generation"])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    records, timings, total_tokens = [], [], 0
    repeats = cfg["benchmark"]["repeats"] if benchmark else 1
    if repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for sample in samples:
            first_tokens = None
            for repeat in range(repeats):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                prediction, token_count, token_ids = predict(model, tokenizer, collator, sample, device, cfg["generation"])
                torch.cuda.synchronize(device)
                latency = time.perf_counter() - start
                timings.append(latency)
                total_tokens += token_count
                if repeat == 0:
                    first_tokens = token_ids
                    record = {"id": sample.id, "level": sample.level, "image_path": sample.image_path,
                              "reference": sample.text, "prediction": prediction, "generated_tokens": token_count,
                              "at_generation_limit": token_count >= cfg["generation"]["max_new_tokens"], "latency_seconds": latency}
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                elif token_ids != first_tokens:
                    raise RuntimeError("Greedy output changed between benchmark repeats")
    snapshot = Path(original.config._name_or_path)
    if not snapshot.is_dir():
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(cfg["model"]["name"], revision=cfg["model"]["revision"], local_files_only=True))
    base_storage = tree_bytes(snapshot, weights_only=True)
    adapter_storage = tree_bytes(cfg["adapter"], weights_only=True) if "adapter" in cfg else 0
    protocol = {"test_fingerprint": audit["test_fingerprint"], "sample_ids": [s.id for s in samples],
                "preprocessing": cfg["preprocessing"], "generation": cfg["generation"], "benchmark": cfg["benchmark"],
                "model": cfg["model"], "gpu_name": torch.cuda.get_device_name(device), "seed": cfg["seed"],
                "software": environment()["packages"], "torch_threads": torch.get_num_threads()}
    results = {"experiment": cfg["experiment"], "status": "smoke" if limit is not None else "completed",
               "protocol": protocol, "protocol_hash": digest(protocol), "quality": grouped_quality(records),
               "inference": {"load_seconds": load_seconds, "load_peak_allocated_bytes": load_peak,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                             "weight_file_bytes": base_storage + adapter_storage, "adapter_weight_bytes": adapter_storage,
                             "warmup_runs": warmup, "measured_runs": len(timings),
                             "latency_mean_seconds": statistics.mean(timings), "latency_p50_seconds": percentile(timings, 0.5),
                             "latency_p95_seconds": percentile(timings, 0.95), "images_per_second": len(timings) / sum(timings),
                             "tokens_per_second": total_tokens / sum(timings), "latencies_seconds": timings,
                             "timing_scope": "batch=1, image IO + preprocessing + H2D + generation + decode; excludes loading/warmup/result writes; OS file cache may be warm"},
               "generation_limit_count": sum(r["at_generation_limit"] for r in records)}
    write_json(output / "metrics.json", results)
    write_json(output / "error_analysis.json", error_analysis(records))
    with (output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        row = {"experiment": cfg["experiment"], "status": results["status"], "cer": results["quality"]["pooled"]["cer"],
               **{k: v for k, v in results["inference"].items() if isinstance(v, (int, float))}}
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps({"experiment": cfg["experiment"], "status": results["status"], "quality": results["quality"]}, indent=2))
    return results


def main():
    p = parser("Evaluate pooled OCR CER and save per-sample predictions")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, help="Smoke subset only; marked as smoke")
    p.add_argument("--adapter", help="Override adapter path, e.g. a smoke checkpoint with --limit")
    p.add_argument("--artifact", help="Override E2a/E2b artifact path; smoke artifacts still require --limit")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.artifact:
        if cfg["experiment"] not in {"E2a", "E2b"}:
            p.error("--artifact applies only to E2a/E2b")
        cfg["artifact"] = args.artifact
    if args.adapter:
        if cfg["experiment"] not in {"E1", "E3"}:
            p.error("--adapter applies only to E1/E3")
        cfg["adapter"] = args.adapter
        if cfg["experiment"] == "E3":
            cfg["base_artifact"] = str(Path(args.adapter).parent / "base_nf4")
    evaluate(cfg, args.output, args.limit)


if __name__ == "__main__":
    main()
