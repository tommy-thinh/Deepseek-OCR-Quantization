"""Plot only compatible completed runs, never smoke or historical results."""
import csv
import json
from pathlib import Path

from .common import new_output, write_json


def collect(paths):
    rows, protocol = [], None
    for path in paths:
        metrics = json.loads(Path(path).read_text())
        if metrics["status"] != "completed":
            raise ValueError(f"{path}: smoke results are not benchmark results")
        if protocol is not None and metrics["protocol_hash"] != protocol:
            raise ValueError("Incompatible inputs, generation, hardware, or preprocessing across runs")
        protocol = metrics["protocol_hash"]
        rows.append({"experiment": metrics["experiment"], "cer": metrics["quality"]["pooled"]["cer"],
                     **{k: v for k, v in metrics["inference"].items() if isinstance(v, (int, float))}})
    if len({r["experiment"] for r in rows}) != len(rows):
        raise ValueError("Duplicate experiment IDs; compare one run per experiment")
    return rows


def main():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--training", nargs="*", default=[], help="Optional E1/E3 training_metrics.json files")
    args = p.parse_args()
    rows = collect(args.metrics)
    out = new_output(args.output)
    write_json(out / "comparison.json", rows)
    with (out / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.training:
        training = []
        training_protocol = None
        for path in args.training:
            result = json.loads(Path(path).read_text())
            if result["smoke"]:
                raise ValueError("Smoke training metrics cannot be compared as completed runs")
            if training_protocol is not None and result["training_protocol_hash"] != training_protocol:
                raise ValueError("Training protocol mismatch between E1 and E3")
            training_protocol = result["training_protocol_hash"]
            training.append({"experiment": result["experiment"], "seconds": result["seconds"],
                             "images_per_second": result["images_per_second"], "checkpoint_bytes": result["checkpoint_bytes"],
                             "adapter_bytes": result["adapter_bytes"], "trainable_parameters": result["trainable_parameters"],
                             "peak_allocated_bytes": max(p["peak_allocated_bytes"] for p in result["peaks_by_rank"])})
        with (out / "training_comparison.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(training[0]))
            writer.writeheader()
            writer.writerows(training)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, key, label in zip(axes, ("seconds", "images_per_second"), ("Training wall time (s)", "Training images/second")):
            ax.bar([row["experiment"] for row in training], [row[key] for row in training])
            ax.set_ylabel(label)
        fig.tight_layout()
        fig.savefig(out / "training_efficiency.png", dpi=200)
        plt.close(fig)
    for key, label in [("peak_allocated_bytes", "Peak allocated GPU memory (GiB)"), ("latency_p50_seconds", "Median end-to-end latency (s)")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        scale = 2 ** 30 if key.endswith("bytes") else 1
        for row in rows:
            ax.scatter(row[key] / scale, row["cer"])
            ax.annotate(row["experiment"], (row[key] / scale, row["cer"]), xytext=(5, 5), textcoords="offset points")
        ax.set(xlabel=label, ylabel="Corpus CER (lower is better)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"cer_vs_{key}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
