import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import yaml


def merge_dict(base, update):
    result = dict(base)
    for key, value in update.items():
        result[key] = merge_dict(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge_dict(load_config(path.parent / parent), cfg)
    expected = {"E0": "bf16", "E1": "bf16", "E2a": "int8", "E2b": "nf4", "E3": "nf4"}
    if "experiment" in cfg and expected.get(cfg["experiment"]) != cfg["precision"]:
        raise ValueError("Experiment ID and precision disagree")
    if not cfg["model"]["revision"] or cfg["model"]["revision"] == "main":
        raise ValueError("Pin an immutable model revision")
    return cfg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def training_settings(settings, world_size):
    batch = settings["per_device_batch_size"]
    effective = settings["effective_batch_size"]
    if world_size < 1 or batch < 1 or effective < 1 or effective % (world_size * batch):
        raise ValueError("Effective batch size must be divisible by GPU count times per-device batch size")
    return {**settings, "expected_world_size": world_size,
            "gradient_accumulation_steps": effective // (world_size * batch)}


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output already contains files: {path}; choose a new run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def environment():
    info = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "unsloth", "jiwer", "Pillow"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    import torch
    info.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    info.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    info["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i), "bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    try:
        info["nvidia_driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        info["nvidia_driver"] = None
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def gpu_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required; this benchmark does not silently fall back to FP16")
    return torch.device("cuda", rank)


def tree_bytes(path, weights_only=False):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and (not weights_only or p.suffix in {".safetensors", ".bin"}))


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    return p
