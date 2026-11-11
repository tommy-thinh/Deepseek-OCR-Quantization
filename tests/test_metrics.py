import pytest

from src.metrics import error_analysis, grouped_quality, normalize, quality


def test_unicode_and_corpus_cer():
    assert normalize("  Tiếng\nViệt  ") == "Tiếng Việt"
    rows = [{"level": "line", "reference": "abc", "prediction": "axc"},
            {"level": "paragraph", "reference": "abcdefghij", "prediction": "abcdefghij"}]
    result = grouped_quality(rows)
    assert result["pooled"]["cer"] == pytest.approx(1 / 13)
    assert result["line"]["cer"] == pytest.approx(1 / 3)


def test_runaway_empty_and_empty_evaluation():
    assert quality([{"reference": "a", "prediction": "aaaa"}])["cer"] == 3
    assert quality([{"reference": "abc", "prediction": ""}])["missing_text_count"] == 1
    with pytest.raises(ValueError):
        quality([])


def test_error_candidates_preserve_granularity():
    report = error_analysis([{"id": "line/test/250/1.jpg", "level": "line", "reference": "a", "prediction": "aaaa"},
                             {"id": "paragraph/test/250/1.jpg", "level": "paragraph", "reference": "abc", "prediction": ""}])
    assert report["worst_by_level"]["line"][0]["over_generation"]
    assert report["worst_by_level"]["line"][0]["high_insertion_candidate"]
    assert report["worst_by_level"]["paragraph"][0]["missing_text"]
