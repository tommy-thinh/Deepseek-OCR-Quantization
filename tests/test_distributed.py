from torch.utils.data import DistributedSampler
import pytest
import torch

from src.common import load_config, training_settings


@pytest.mark.parametrize("world,accumulation", [(1, 16), (2, 8), (4, 4)])
def test_effective_batch_is_preserved(world, accumulation):
    settings = load_config("configs/E1.yaml")["train"]
    resolved = training_settings(settings, world)
    assert resolved["gradient_accumulation_steps"] == accumulation
    assert resolved["expected_world_size"] == world
    assert world * resolved["per_device_batch_size"] * accumulation == 16
    assert "expected_world_size" not in settings


def test_invalid_gpu_count_rejected():
    settings = load_config("configs/E1.yaml")["train"]
    for world in (0, 3, 32):
        with pytest.raises(ValueError, match="divisible"):
            training_settings(settings, world)


def test_single_gpu_accumulation_matches_full_batch_update():
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1)
    accumulated = torch.nn.Linear(3, 1)
    accumulated.load_state_dict(model.state_dict())
    x, y = torch.randn(16, 3), torch.randn(16, 1)
    full_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    micro_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    torch.nn.functional.mse_loss(model(x), y).backward()
    for i in range(16):
        (torch.nn.functional.mse_loss(accumulated(x[i:i+1]), y[i:i+1]) / 16).backward()
    full_optimizer.step()
    micro_optimizer.step()
    for full, micro in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(full, micro)


def test_ddp_sharding_covers_all_samples_and_records_padding():
    samples = list(range(7))
    shards = [list(DistributedSampler(samples, num_replicas=2, rank=r, shuffle=False)) for r in range(2)]
    assert len(shards[0]) == len(shards[1]) == 4
    assert set(shards[0] + shards[1]) == set(samples)
    assert len(set(shards[0]) & set(shards[1])) == 1
