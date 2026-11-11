import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from src.merge import verified_merge


def test_merge_checks_every_target_and_removes_adapters():
    base = GPT2LMHeadModel(GPT2Config(n_embd=8, n_layer=1, n_head=1, vocab_size=16)).to(torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True))
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    merged, report = verified_merge(model)
    assert report["all_weight_hashes_match"] and report["layers_checked"] == 1
    assert not any("lora_" in name for name, _ in merged.named_parameters())
