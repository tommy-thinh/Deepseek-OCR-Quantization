"""The pinned upstream image transform/crop helpers with one shared OCR prefix."""
import importlib
import math

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence


class OCRCollator:
    def __init__(self, tokenizer, helpers, config):
        self.tokenizer, self.helpers, self.cfg = tokenizer, helpers, config
        if (config["base_size"], config["image_size"], config["crop_mode"]) != (1024, 640, True):
            raise ValueError("Only the upstream 1024/640 crop-mode protocol has been validated")
        if config["prompt"].count("<image>") != 1:
            raise ValueError("Each sample requires exactly one <image> marker")
        if tokenizer.convert_tokens_to_ids("<image>") != 128815:
            raise ValueError("Unexpected DeepSeek-OCR image token ID")
        self.transform = helpers.BasicImageTransform(mean=(0.5,) * 3, std=(0.5,) * 3, normalize=True)

    @classmethod
    def from_model(cls, model, tokenizer, config):
        return cls(tokenizer, importlib.import_module(type(model).__module__), config)

    def encode(self, sample, training=True):
        with Image.open(sample.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        tiles, ratio = [], (1, 1)
        if max(image.size) > 640:
            tiles, ratio = self.helpers.dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False)
        original = self.transform(ImageOps.pad(image, (1024, 1024), color=(127, 127, 127))).unsqueeze(0).to(torch.bfloat16)
        width, height = ratio
        crops = torch.stack([self.transform(tile) for tile in tiles]).to(torch.bfloat16) if width > 1 or height > 1 else torch.zeros((1, 3, 1024, 1024), dtype=torch.bfloat16)
        n, base = math.ceil(640 / 64), math.ceil(1024 / 64)
        image_tokens = [128815] * (base * (base + 1) + 1)
        if width > 1 or height > 1:
            image_tokens += [128815] * (height * n * (width * n + 1))
        # Match upstream infer(): plain template, explicit BOS, expanded image marker.
        prompt = self.helpers.format_messages([
            {"role": "<|User|>", "content": self.cfg["prompt"]},
            {"role": "<|Assistant|>", "content": ""},
        ], sft_format="plain", system_prompt="")
        before, after = prompt.split("<image>")
        left = [self.tokenizer.bos_token_id] + self.tokenizer.encode(before, add_special_tokens=False)
        right = self.tokenizer.encode(after, add_special_tokens=False)
        prefix = left + image_tokens + right
        target = self.tokenizer.encode(sample.text, add_special_tokens=False) + [self.tokenizer.eos_token_id] if training else []
        ids = prefix + target
        if len(ids) > self.cfg["max_length"]:
            raise ValueError(f"{sample.id}: {len(ids)} tokens exceed max_length; refuse to truncate images or reference")
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor([-100] * len(prefix) + target),
                "images_seq_mask": torch.tensor([False] * len(left) + [True] * len(image_tokens) + [False] * (len(right) + len(target))),
                "images": (crops, original), "images_spatial_crop": torch.tensor([[width, height]])}

    def __call__(self, samples, training=True):
        if not training and len(samples) != 1:
            raise ValueError("Inference protocol requires batch size one")
        rows = [self.encode(sample, training=training) for sample in samples]
        ids = pad_sequence([row["input_ids"] for row in rows], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # Build attention from lengths: PAD can alias EOS and EOS still needs supervision.
        attention = torch.arange(ids.shape[1])[None, :] < torch.tensor([len(row["input_ids"]) for row in rows])[:, None]
        batch = {"input_ids": ids, "attention_mask": attention.long(),
                 "images_seq_mask": pad_sequence([row["images_seq_mask"] for row in rows], batch_first=True, padding_value=False),
                 "images": [row["images"] for row in rows],
                 "images_spatial_crop": torch.cat([row["images_spatial_crop"] for row in rows])}
        if training:
            batch["labels"] = pad_sequence([row["labels"] for row in rows], batch_first=True, padding_value=-100)
        return batch


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value
