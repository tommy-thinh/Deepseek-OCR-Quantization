import importlib

import pytest


@pytest.mark.parametrize("module", ["data", "preprocessing", "model", "train", "quantize", "evaluate", "benchmark"])
def test_pipeline_imports(module):
    importlib.import_module(f"src.{module}")
