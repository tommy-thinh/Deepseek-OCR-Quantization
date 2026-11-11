"""Validate the actual PEFT 0.17.1 BF16 merge arithmetic, independently of decoding."""
import hashlib

import torch


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def verified_merge(model):
    expected = {}
    for name, layer in model.named_modules():
        if not hasattr(layer, "lora_A"):
            continue
        if set(layer.lora_A) != {"default"} or layer.lora_variant or layer.lora_bias["default"]:
            raise ValueError("Merge audit supports one vanilla, bias-free LoRA adapter only")
        weight = layer.get_base_layer().weight.detach()
        # Match pinned PEFT safe_merge: cast delta to base dtype, then add in base dtype.
        # https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/layer.py
        merged_weight = weight.clone()
        merged_weight += layer.get_delta_weight("default").detach().to(weight.dtype)
        if not torch.isfinite(merged_weight).all():
            raise ValueError(f"Non-finite merged weight: {name}")
        expected[name.removeprefix("base_model.model.")] = tensor_hash(merged_weight)
    if not expected:
        raise ValueError("No LoRA layers to merge")
    merged = model.merge_and_unload(safe_merge=True)
    mismatches = [name for name, checksum in expected.items() if tensor_hash(merged.get_submodule(name).weight) != checksum]
    if mismatches:
        raise RuntimeError(f"Merged weights differ from independently computed LoRA sums: {mismatches[:5]}")
    return merged, {"layers_checked": len(expected), "all_weight_hashes_match": True,
                    "criterion": "Every target weight exactly equals the pinned PEFT base-dtype W + cast(scale*B@A) arithmetic"}
