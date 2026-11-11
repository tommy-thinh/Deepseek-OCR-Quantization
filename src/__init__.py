"""Reproducible DeepSeek-OCR benchmark; run commands with python -m src.<module>."""
import os
from pathlib import Path

# Match torchrun's default thread budget for direct Python launches.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Keep downloaded code/weights inside the project unless the caller sets a cache.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / ".cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
