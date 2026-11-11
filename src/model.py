"""One upstream model/revision for all five experiments (no Unsloth monkey patches)."""
import json
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .common import training_settings


def quantization_config(precision, skip_modules):
    # https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes
    if precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=skip_modules)
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                                  llm_int8_has_fp16_weight=False, llm_int8_skip_modules=skip_modules)
    if precision != "bf16":
        raise ValueError(precision)
    return None


def load_base(cfg, device, source=None, precision=None):
    source = source or cfg["model"]["name"]
    precision = precision or cfg["precision"]
    revision = {} if Path(source).is_dir() else {"revision": cfg["model"]["revision"], "code_revision": cfg["model"]["revision"]}
    kwargs = dict(trust_remote_code=True, dtype=torch.bfloat16,
                  attn_implementation=cfg["model"]["attention"], **revision)
    quant = quantization_config(precision, cfg["model"]["skip_modules"])
    if quant:
        kwargs.update(quantization_config=quant, device_map={"": str(device)})
    model = AutoModel.from_pretrained(source, **kwargs)
    if not quant:
        model = model.to(device)
    positions = model.model.vision_model.embeddings.position_ids
    if not torch.equal(positions.cpu(), torch.arange(positions.numel()).reshape(positions.shape)):
        raise RuntimeError("The upstream positional-ID buffer was not initialized correctly")
    converted = [(name, module) for name, module in model.named_modules() if type(module).__name__ in {"Linear4bit", "Linear8bitLt"}]
    if quant:
        expected_class = "Linear4bit" if precision == "nf4" else "Linear8bitLt"
        if not converted or any(type(module).__name__ != expected_class for _, module in converted):
            raise RuntimeError("Requested quantization format was not applied as configured")
        if any(any(skip in name.split(".") for skip in cfg["model"]["skip_modules"]) for name, _ in converted):
            raise RuntimeError("A protected vision/projector/output module was quantized")
        if precision == "nf4" and any(module.weight.quant_type != "nf4" for _, module in converted):
            raise RuntimeError("A 4-bit module is not NF4")
    elif converted:
        raise RuntimeError("BF16 experiment unexpectedly contains quantized modules")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, **revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def attach_lora(model, cfg):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Limit targets to the decoder, even if a future vision module has the same suffix.
    targets = [name for name, module in model.named_modules()
               if name.startswith("model.layers.") and name.rsplit(".", 1)[-1] in cfg["lora"]["target_modules"]]
    if not targets:
        raise RuntimeError("No decoder LoRA targets found; inspect the pinned model implementation")
    options = {k: v for k, v in cfg["lora"].items() if k != "target_modules"}
    model = get_peft_model(model, LoraConfig(**options, target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    # Keep adapter parameters/AdamW moments FP32 in BOTH E1 and E3, BF16 autocast for matmuls.
    # Non-reentrant checkpointing supports frozen inputs; prepare_model_for_kbit_training
    # would upcast the entire frozen vision tower, changing the comparison's memory cost.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable parameter: {name}")
            parameter.data = parameter.data.float()
    return model


def validate_adapter(path, cfg, experiment, allow_smoke=False):
    path = Path(path)
    if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"No complete adapter checkpoint at {path}")
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Adapter checkpoint has no completion marker: {path}")
    metadata = json.loads((path / "provenance.json").read_text())
    if metadata["experiment"] != experiment or metadata["model"] != cfg["model"]:
        raise ValueError("Adapter experiment or base model/revision differs")
    if metadata["lora"] != cfg["lora"] or metadata["preprocessing"] != cfg["preprocessing"]:
        raise ValueError("Adapter LoRA/preprocessing protocol differs")
    if metadata["train"] != training_settings(cfg["train"], metadata["world_size"]) or metadata["seed"] != cfg["seed"]:
        raise ValueError("Adapter training settings or seed differ from configuration")
    if metadata.get("smoke", False) and not allow_smoke:
        raise ValueError("Smoke adapters cannot be used as completed benchmark checkpoints")
    return metadata


def load_experiment(cfg, device, allow_smoke=False):
    experiment = cfg["experiment"]
    if experiment in {"E2a", "E2b"}:
        source = Path(cfg["artifact"])
        manifest = json.loads((source / "quantization_provenance.json").read_text())
        if manifest["experiment"] != experiment or manifest["model"] != cfg["model"]:
            raise ValueError("Quantized artifact provenance mismatch")
        if manifest["precision"] != cfg["precision"] or manifest["skipped_modules"] != cfg["model"]["skip_modules"]:
            raise ValueError("Quantized artifact precision/module policy differs")
        if manifest.get("smoke", False) and not allow_smoke:
            raise ValueError("Smoke PTQ artifacts cannot be evaluated as completed experiments")
        model, tokenizer = load_base(cfg, device, source=str(source))
    else:
        source = None
        if experiment == "E3":
            source = cfg["base_artifact"]
            manifest = json.loads((Path(source) / "base_provenance.json").read_text())
            if manifest != {"model": cfg["model"], "precision": "nf4", "source": "original_pretrained"}:
                raise ValueError("E3 must use the original pretrained NF4 base")
        model, tokenizer = load_base(cfg, device, source=source)
        if experiment in {"E1", "E3"}:
            validate_adapter(cfg["adapter"], cfg, experiment, allow_smoke=allow_smoke)
            model = PeftModel.from_pretrained(model, cfg["adapter"], is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def precision_report(model, optimizer=None):
    parameters = []
    for name, parameter in model.named_parameters():
        parameters.append({"name": name, "class": type(parameter).__name__, "dtype": str(parameter.dtype),
                           "shape": list(parameter.shape), "numel": parameter.numel(),
                           "bytes": parameter.numel() * parameter.element_size(), "trainable": parameter.requires_grad})
    quantized = []
    for name, module in model.named_modules():
        if type(module).__name__ in {"Linear4bit", "Linear8bitLt"}:
            state = getattr(module.weight, "quant_state", None)
            quantized.append({"name": name, "class": type(module).__name__,
                              "weight_storage_dtype": str(module.weight.dtype),
                              "compute_dtype": str(getattr(module, "compute_dtype", "LLM.int8 mixed kernels")),
                              "quant_type": getattr(getattr(module, "weight", None), "quant_type", None),
                              "scale_dtype": str(getattr(getattr(state, "absmax", None), "dtype", None)),
                              "nested_scale_dtype": str(getattr(getattr(getattr(state, "state2", None), "absmax", None), "dtype", None))})
    optimizer_dtypes = Counter()
    if optimizer:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_dtypes[f"{key}/{value.dtype}"] += value.numel()
    return {"compute_policy": "BF16 autocast; INT8 uses bitsandbytes LLM.int8 mixed kernels",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_storage_bytes": sum(row["bytes"] for row in parameters),
            "parameters": parameters, "quantized_modules": quantized,
            "buffers": [{"name": n, "dtype": str(b.dtype), "bytes": b.numel() * b.element_size()} for n, b in model.named_buffers()],
            "optimizer_states": dict(optimizer_dtypes),
            "storage_note": "Parameter totals exclude some bitsandbytes side statistics; use serialized weight size and CUDA peaks as separate measurements."}
