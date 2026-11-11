"""Regenerate README figures and analysis tables from saved results; no GPU/model needed."""
import csv
import argparse
import json
import statistics
import textwrap
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
ANALYSIS = ROOT / "results/analysis"
IDS = ["E0", "E1", "E2a", "E2b", "E3"]
COLORS = {"E0": "#777777", "E1": "#0072B2", "E1-M": "#56B4E9", "E2a": "#009E73", "E2b": "#D55E00", "E3": "#CC79A7"}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def load_results():
    metrics, records, diagnostics = {}, {}, {}
    for experiment in IDS:
        folder = f"results/standard/{experiment}"
        m = read(f"{folder}/metrics.json")
        assert m["status"] == "completed" and m["quality"]["pooled"]["samples"] == 232
        metrics[experiment] = m
        records[experiment] = {r["id"]: r for r in map(json.loads, (ROOT / folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        diagnostics[experiment] = {r["id"]: r for r in read(f"{folder}/error_analysis.json")["samples"]}
        assert set(records[experiment]) == set(diagnostics[experiment]) == set(m["protocol"]["sample_ids"])
        for level in ("line", "paragraph"):
            rows = [r for r in diagnostics[experiment].values() if r["level"] == level]
            assert len(rows) == m["quality"][level]["samples"]
            errors = sum(round(r["cer"] * r["reference_characters"]) for r in rows)
            q = m["quality"][level]
            assert errors == q["substitutions"] + q["deletions"] + q["insertions"]
            assert abs(errors / sum(r["reference_characters"] for r in rows) - q["cer"]) < 1e-12
        for sample_id, record in records[experiment].items():
            d = diagnostics[experiment][sample_id]
            assert len(normalize(record["reference"])) == d["reference_characters"]
            assert len(normalize(record["prediction"])) == d["prediction_characters"]
            assert record["at_generation_limit"] == d["at_generation_limit"]
    assert len({m["protocol_hash"] for m in metrics.values()}) == 1
    for experiment in IDS:
        assert set(records[experiment]) == set(records["E1"])
        assert all(r["reference"] == records["E1"][sid]["reference"] for sid, r in records[experiment].items())
    merge = read("results/quantization/E1/merge_verification.json")
    assert merge["passed"] and merge["weight_audit"]["all_weight_hashes_match"]
    assert len(merge["sample_ids"]) == 232 and merge["unmerged_quality"] == metrics["E1"]["quality"]
    return metrics, records, diagnostics, merge


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bars(ax, names, values, title, ylabel):
    bars = ax.bar(names, values, color=[COLORS[n] for n in names], width=0.65)
    for name, bar, value in zip(names, bars, values):
        if name == "E1-M":
            bar.set_hatch("//")
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.18))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)


def quality_figure(metrics, merge):
    names = ["E1", "E1-M", "E2a", "E2b", "E3"]
    qualities = {e: m["quality"] for e, m in metrics.items()}
    qualities["E1-M"] = merge["merged_bf16_quality"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    bars(axes[0], names, [100 * qualities[e]["line"]["cer"] for e in names], "Lines: fine-tuned variants (201 images)", "Corpus CER (%) — lower is better")
    names = ["E0", *names]
    bars(axes[1], names, [100 * qualities[e]["paragraph"]["cer"] for e in names], "Paragraphs: all variants (31 images)", "Corpus CER (%) — lower is better")
    fig.suptitle("Small line-CER differences, large paragraph tail failures", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    fig.text(0.5, 0.015, "E1-M = merged BF16 control, before PTQ. E0 line CER = 93.64% (omitted from the line panel).\nPanels use different scales. CER can exceed 100% when generated text contains many insertions.", ha="center", fontsize=9)
    save(fig, "quality_by_granularity")


def paragraph_figure(metrics, diagnostics, merge):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    names = ["E0", "E1", "E1-M", "E2a", "E2b", "E3"]
    q = {e: m["quality"]["paragraph"] for e, m in metrics.items()}
    q["E1-M"] = merge["merged_bf16_quality"]["paragraph"]
    chars = sum(r["reference_characters"] for r in diagnostics["E1"].values() if r["level"] == "paragraph")
    bottom = np.zeros(len(names))
    for key, label, color in [("substitutions", "Substitutions", "#0072B2"), ("deletions", "Deletions", "#E69F00"), ("insertions", "Insertions", "#D55E00")]:
        values = np.array([100 * q[e][key] / chars for e in names])
        axes[0].bar(names, values, bottom=bottom, label=label, color=color, width=0.65)
        bottom += values
    axes[0].set(title="Paragraph error composition", ylabel="Edits / reference characters (%)", ylim=(0, 178))
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.18)
    rng = np.random.default_rng(42)
    for index, experiment in enumerate(IDS):
        values = [100 * r["cer"] for r in diagnostics[experiment].values() if r["level"] == "paragraph"]
        axes[1].scatter(index + rng.uniform(-0.13, 0.13, len(values)), values, color=COLORS[experiment], alpha=0.75, s=22)
        axes[1].scatter(index, statistics.median(values), marker="D", color="black", s=28, zorder=5)
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_xticks(range(len(IDS)), IDS)
    axes[1].set_yticks([0, 5, 10, 100, 1000], ["0", "5", "10", "100", "1000"])
    axes[1].set(title="Each dot is one paragraph; diamonds are medians", ylabel="Per-paragraph CER (%) — nonlinear scale")
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("A few runaway outputs dominate paragraph CER", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "paragraph_failures")


def resources_figure(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title in zip(axes, ["weight_file_bytes", "peak_allocated_bytes"], ["Serialized base + adapter where applicable", "Peak allocated CUDA memory during inference"]):
        values = [metrics[e]["inference"][key] / 2**30 for e in IDS]
        bars(ax, IDS, values, title, "GiB")
    fig.suptitle("Quantization reduces serialized size and allocated GPU memory", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "storage_memory")


def example_figure(records, diagnostics):
    sid = "paragraph/test/253/4.jpg"
    reference = records["E1"][sid]["reference"]
    fig = plt.figure(figsize=(11, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.28, wspace=0.12)
    ax = fig.add_subplot(grid[0, :])
    with Image.open(ROOT / records["E1"][sid]["image_path"]) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(f"Same input: {sid} — {len(normalize(reference))} reference characters", fontsize=11)
    ax.axis("off")
    for index, experiment in enumerate(["E1", "E2b"]):
        record = records[experiment][sid]
        d = diagnostics[experiment][sid]
        ax = fig.add_subplot(grid[1, index]); ax.axis("off")
        title = f"{experiment}: {d['prediction_characters']:,} characters | CER {100*d['cer']:.2f}%"
        excerpt = "Start: " + normalize(record["prediction"])[:155] + "…"
        tail = "End: …" + normalize(record["prediction"])[-115:]
        footer = f"{record['generated_tokens']} generated tokens" + (" — hit 2,048-token cap" if record["at_generation_limit"] else " — stopped before the cap")
        ax.text(0, 1, title, transform=ax.transAxes, va="top", weight="bold", color=COLORS[experiment], fontsize=11)
        ax.text(0, 0.82, textwrap.fill(excerpt, 61) + "\n\n" + textwrap.fill(tail, 61), transform=ax.transAxes, va="top", fontsize=9.5, linespacing=1.4)
        ax.text(0, 0.02, footer, transform=ax.transAxes, va="bottom", fontsize=9, weight="bold")
    fig.suptitle("Recognition errors versus repetitive over-generation", fontsize=14, fontweight="bold", y=0.98)
    save(fig, "paragraph_example")


def main():
    global OUT, ANALYSIS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--analysis-output", type=Path, default=ANALYSIS)
    args = parser.parse_args()
    OUT, ANALYSIS = args.output, args.analysis_output
    OUT.mkdir(exist_ok=True); ANALYSIS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    metrics, records, diagnostics, merge = load_results()
    summary, paragraph = [], []
    for e in IDS:
        m = metrics[e]
        rows = [r for r in diagnostics[e].values() if r["level"] == "paragraph"]
        total_errors = sum(round(r["cer"] * r["reference_characters"]) for r in rows)
        cap_errors = sum(round(r["cer"] * r["reference_characters"]) for r in rows if r["at_generation_limit"])
        summary.append({"experiment": e, **{f"{level}_cer_pct": m["quality"][level]["cer"] * 100 for level in ["line", "paragraph", "pooled"]},
                        "recorded_mean_latency_seconds": m["inference"]["latency_mean_seconds"],
                        "weight_GiB": m["inference"]["weight_file_bytes"] / 2**30, "peak_allocated_GiB": m["inference"]["peak_allocated_bytes"] / 2**30,
                        "paragraph_median_cer_pct": statistics.median(r["cer"] for r in rows) * 100,
                        "paragraph_capped": sum(r["at_generation_limit"] for r in rows), "capped_paragraph_error_share_pct": 100 * cap_errors / total_errors})
        paragraph.extend({"experiment": e, **r, "character_edit_errors": round(r["cer"] * r["reference_characters"])} for r in rows)
    for name, rows in [("summary.csv", summary), ("paragraph_errors.csv", paragraph)]:
        with (ANALYSIS / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    quality_figure(metrics, merge);paragraph_figure(metrics, diagnostics, merge)
    resources_figure(metrics);example_figure(records, diagnostics)
    print("Generated four PNG/PDF figures, summary.csv and paragraph_errors.csv; saved metrics/predictions reconciled exactly.")


if __name__ == "__main__":
    main()
