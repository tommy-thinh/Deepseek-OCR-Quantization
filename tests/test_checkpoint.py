import json

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from src.checkpoint import protocol, save_checkpoint, validate_resume
from src.common import load_config, training_settings
from src.model import quantization_config, validate_adapter


class TokenizerStub:
    def save_pretrained(self, path):
        (path / "tokenizer_config.json").write_text("{}")


def test_checkpoint_roundtrip_and_protocol_guard(tmp_path):
    torch.manual_seed(42)
    base = GPT2LMHeadModel(GPT2Config(n_embd=8, n_layer=1, n_head=1, vocab_size=16))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ids = torch.tensor([[1, 2, 3]])
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    model.eval()
    before = model(input_ids=ids).logits.detach()
    # Same original base weights, adapter reload must recover trained predictions.
    torch.manual_seed(42)
    original = GPT2LMHeadModel(base.config)
    path = tmp_path / "checkpoint-1"
    metadata = {"experiment": "E1", "seed": 42}
    save_checkpoint(path, model, TokenizerStub(), {"optimizer": optimizer.state_dict(), "step": 1}, metadata)
    assert validate_resume(path, metadata) == path
    restored = PeftModel.from_pretrained(original, path).eval()
    assert torch.allclose(before, restored(input_ids=ids).logits)
    state = torch.load(path / "training_state.pt", weights_only=False)
    assert state["step"] == 1 and state["optimizer"]["state"]
    with pytest.raises(ValueError, match="protocol differs"):
        validate_resume(path, {"experiment": "E3", "seed": 42})
    (path / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_resume(path, metadata)


def test_experiment_matrix_and_explicit_nf4():
    for experiment, precision in [("E0", "bf16"), ("E1", "bf16"), ("E2a", "int8"), ("E2b", "nf4"), ("E3", "nf4")]:
        cfg = load_config(f"configs/{experiment}.yaml")
        assert cfg["precision"] == precision
        assert cfg["train"]["effective_batch_size"] == 16
    config = quantization_config("nf4", ["vision_model"])
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert config.bnb_4bit_use_double_quant
    assert config.llm_int8_skip_modules == ["vision_model"]
    assert quantization_config("int8", []).load_in_8bit


@pytest.mark.parametrize("world", [1, 2])
def test_adapter_hardware_settings_and_resume_guard(tmp_path, world):
    cfg = load_config("configs/E1.yaml")
    trained = {**cfg, "train": training_settings(cfg["train"], world)}
    metadata = protocol(trained, {"train_fingerprint": "train", "test_fingerprint": "test"}, world)
    for name in ("adapter_config.json", "adapter_model.safetensors", "training_state.pt", "COMPLETE"):
        (tmp_path / name).touch()
    (tmp_path / "provenance.json").write_text(json.dumps(metadata))
    assert validate_adapter(tmp_path, cfg, "E1") == metadata
    assert validate_resume(tmp_path, metadata) == tmp_path
    with pytest.raises(ValueError, match="GPU count"):
        validate_resume(tmp_path, {**metadata, "world_size": 3 - world})
    changed = {**cfg, "train": {**cfg["train"], "learning_rate": 0.01}}
    with pytest.raises(ValueError, match="training settings"):
        validate_adapter(tmp_path, changed, "E1")
