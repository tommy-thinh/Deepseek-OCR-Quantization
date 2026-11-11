import unicodedata

from jiwer import process_characters


NORMALIZATION = "Unicode NFC; collapse whitespace to one space; strip; preserve case/punctuation/diacritics"


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def quality(records):
    if not records:
        raise ValueError("Cannot score an empty evaluation")
    references = [normalize(r["reference"]) for r in records]
    predictions = [normalize(r["prediction"]) for r in records]
    details = process_characters(references, predictions)
    return {"samples": len(records), "cer": details.cer,
            "substitutions": details.substitutions, "deletions": details.deletions, "insertions": details.insertions,
            "over_generation_count": sum(len(p) > 2 * max(len(r), 1) for r, p in zip(references, predictions)),
            "missing_text_count": sum(len(p) < 0.5 * len(r) for r, p in zip(references, predictions)),
            "normalization": NORMALIZATION,
            "diagnostic_note": "Length ratios flag possible over-generation/missing text. Hallucinations require inspecting saved predictions against images."}


def grouped_quality(records):
    result = {"pooled": quality(records)}
    for level in sorted({r["level"] for r in records}):
        result[level] = quality([r for r in records if r["level"] == level])
    return result


def error_analysis(records):
    rows = []
    for record in records:
        reference, prediction = normalize(record["reference"]), normalize(record["prediction"])
        edits = process_characters(reference, prediction)
        rows.append({"id": record["id"], "level": record["level"], "cer": edits.cer,
                     "reference_characters": len(reference), "prediction_characters": len(prediction),
                     "length_ratio": len(prediction) / max(1, len(reference)),
                     "missing_text": len(prediction) < 0.5 * len(reference),
                     "over_generation": len(prediction) > 2 * max(1, len(reference)),
                     "high_insertion_candidate": edits.insertions > 0.5 * max(1, len(reference)),
                     "at_generation_limit": record.get("at_generation_limit", False)})
    return {"samples": rows, "worst_by_level": {level: sorted([r for r in rows if r["level"] == level], key=lambda r: r["cer"], reverse=True)[:20]
                                                  for level in sorted({r["level"] for r in rows})},
            "interpretation": "High insertions/length flags are candidates for image-grounded hallucination review, not verified hallucination labels. Inspect predictions.jsonl and original images."}
