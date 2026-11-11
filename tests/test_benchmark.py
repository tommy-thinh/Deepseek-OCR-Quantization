import json

import pytest

from src.benchmark import collect
from src.evaluate import percentile


def test_percentiles():
    assert percentile([4, 1, 2, 3], 0.5) == 2.5
    assert percentile([1], 0.95) == 1


def test_reject_smoke_and_unfair_comparisons(tmp_path):
    def metric(name, status, protocol):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"experiment": name, "status": status, "protocol_hash": protocol,
                                    "quality": {"pooled": {"cer": 0.1}}, "inference": {"load_seconds": 1}}))
        return path
    a = metric("E0", "completed", "same")
    b = metric("E1", "smoke", "same")
    with pytest.raises(ValueError, match="smoke"):
        collect([a, b])
    b = metric("E1", "completed", "different")
    with pytest.raises(ValueError, match="Incompatible"):
        collect([a, b])


def test_csv_and_plots_use_compatible_completed_fixtures(tmp_path, monkeypatch):
    from PIL import Image
    from src.benchmark import main
    inputs = []
    for i, name in enumerate(("E0", "E1")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"experiment": name, "status": "completed", "protocol_hash": "fixture",
                                    "quality": {"pooled": {"cer": 0.2 - i * 0.01}},
                                    "inference": {"peak_allocated_bytes": 10**9, "latency_p50_seconds": 1 + i}}))
        inputs.append(str(path))
    output = tmp_path / "plots"
    monkeypatch.setattr("sys.argv", ["benchmark", *inputs, "--output", str(output)])
    main()
    assert "E0" in (output / "comparison.csv").read_text()
    for path in output.glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 1000 and image.height >= 700
