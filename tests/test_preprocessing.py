from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.common import load_config
from src.data import Sample
from src.preprocessing import OCRCollator, to_device


class Tokenizer:
    bos_token_id = 0
    eos_token_id = pad_token_id = 1

    def convert_tokens_to_ids(self, token):
        return 128815

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]


def collator():
    helpers = SimpleNamespace(
        BasicImageTransform=lambda **kw: lambda im: torch.ones(3, im.height, im.width),
        dynamic_preprocess=lambda image, **kw: ([image.resize((640, 640))] * 2, (2, 1)),
        format_messages=lambda messages, **kw: messages[0]["content"].strip(),
    )
    return OCRCollator(Tokenizer(), helpers, load_config("configs/E0.yaml")["preprocessing"])


def sample(tmp_path, text="Việt", size=(24, 12)):
    path = tmp_path / f"{len(text)}.png"
    Image.new("RGB", size, "white").save(path)
    return Sample("line/train/1/1.png", "line", "train", "1", str(path), text)


def test_masking_eos_and_padding(tmp_path):
    c = collator()
    short, long = sample(tmp_path, "Có"), sample(tmp_path, "Tiếng Việt")
    batch = c([short, long])
    for i, text in enumerate((short.text, long.text)):
        supervised = batch["labels"][i][batch["labels"][i] != -100].tolist()
        assert supervised == c.tokenizer.encode(text) + [1]
    assert (batch["labels"][batch["images_seq_mask"]] == -100).all()
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert batch["images_seq_mask"].sum(dim=1).tolist() == [273, 273]
    inference = c([short], training=False)
    assert "labels" not in inference
    prefix = c.encode(short)["labels"].eq(-100).sum().item()
    assert inference["input_ids"][0].tolist() == batch["input_ids"][0, :prefix].tolist()


def test_crop_tokens_and_length_guard(tmp_path):
    c = collator()
    row = c.encode(sample(tmp_path, size=(900, 300)))
    assert row["images_seq_mask"].sum().item() == 273 + 210
    assert row["images"][0].shape == (2, 3, 640, 640)
    assert row["images_spatial_crop"].tolist() == [[2, 1]]
    c.cfg["max_length"] = 10
    with pytest.raises(ValueError, match="refuse to truncate"):
        c.encode(sample(tmp_path))


def test_nested_device_transfer(tmp_path):
    batch = collator()([sample(tmp_path)])
    moved = to_device(batch, torch.device("cpu"))
    assert isinstance(moved["images"], list) and isinstance(moved["images"][0], tuple)
    assert moved["images_seq_mask"].dtype == torch.bool
