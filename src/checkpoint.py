import json
from pathlib import Path

from .common import write_json


def protocol(cfg, audit, world_size, smoke=False):
    return {"experiment": cfg["experiment"], "model": cfg["model"], "lora": cfg["lora"],
            "preprocessing": cfg["preprocessing"], "train": cfg["train"], "seed": cfg["seed"],
            "train_fingerprint": audit["train_fingerprint"], "test_fingerprint": audit["test_fingerprint"],
            "world_size": world_size, "smoke": smoke}


def validate_resume(path, expected):
    path = Path(path)
    actual = json.loads((path / "provenance.json").read_text())
    if actual != expected:
        raise ValueError("Checkpoint protocol differs (data, seed, model, GPU count, batch size, or training settings)")
    for file in ("adapter_config.json", "adapter_model.safetensors", "training_state.pt", "COMPLETE"):
        if not (path / file).is_file():
            raise ValueError(f"Incomplete checkpoint: {path / file}")
    return path


def save_checkpoint(path, model, tokenizer, state, metadata):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    staging = path.with_name(path.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    torch.save(state, staging / "training_state.pt")
    write_json(staging / "provenance.json", metadata)
    (staging / "COMPLETE").write_text("complete\n")
    staging.rename(path)
