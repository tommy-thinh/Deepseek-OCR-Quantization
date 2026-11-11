"""LoRA/QLoRA training on one GPU, with optional DDP."""
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

from .checkpoint import protocol, save_checkpoint, validate_resume
from .common import digest, environment, gpu_device, load_config, new_output, parser, seed_all, training_settings, tree_bytes, write_json
from .data import index_split, require_audit
from .model import attach_lora, load_base, precision_report
from .preprocessing import OCRCollator, to_device


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def main():
    p = parser("Train E1 BF16 LoRA or E3 NF4 QLoRA on one GPU or with torchrun DDP")
    p.add_argument("--output", help="New run directory; defaults to models/<experiment>")
    p.add_argument("--resume", help="Complete checkpoint directory")
    p.add_argument("--smoke", action="store_true", help="Two optimizer steps, isolated artifacts; never a benchmark result")
    p.add_argument("--smoke-steps", type=int, choices=[1, 2], default=2, help="Use one step to test checkpoint/resume, then resume to step two")
    args = p.parse_args()
    cfg = load_config(args.config)
    if cfg["experiment"] not in {"E1", "E3"}:
        p.error("Only E1 and E3 train adapters")
    device = gpu_device()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    t = cfg["train"] = training_settings(cfg["train"], world)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    seed_all(cfg["seed"])
    output = Path(args.output or (f"models/smoke-{cfg['experiment']}" if args.smoke else f"models/{cfg['experiment']}"))
    if rank == 0:
        if args.resume:
            output.mkdir(parents=True, exist_ok=True)
        else:
            new_output(output)
    if world > 1:
        dist.barrier()
    audit = require_audit(cfg)
    samples = index_split(cfg["data"]["roots"], "train", allow_unlabeled=cfg["data"]["unlabeled_policy"] == "exclude")
    metadata = protocol(cfg, audit, world, args.smoke)
    metadata["full_data_integrity_passed"] = audit["integrity_passed"]
    if args.resume:
        validate_resume(args.resume, metadata)
    base, tokenizer = load_base(cfg, device)
    collator = OCRCollator.from_model(base, tokenizer, cfg["preprocessing"])
    if cfg["experiment"] == "E3" and rank == 0 and not (output / "base_nf4").exists():
        base_dir = output / "base_nf4"
        base.save_pretrained(base_dir, safe_serialization=True)
        tokenizer.save_pretrained(base_dir)
        write_json(base_dir / "base_provenance.json", {"model": cfg["model"], "precision": "nf4", "source": "original_pretrained"})
    if world > 1:
        dist.barrier()
    model = attach_lora(base, cfg)
    probe_name, probe = next((n, p) for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad)
    if args.resume:
        set_peft_model_state_dict(model, load_file(str(Path(args.resume) / "adapter_model.safetensors")))
    sampler = DistributedSampler(samples, num_replicas=world, rank=rank, seed=cfg["seed"], shuffle=True, drop_last=False)
    loader = DataLoader(samples, batch_size=t["per_device_batch_size"], sampler=sampler, collate_fn=collator, num_workers=0)
    accumulation = t["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = steps_per_epoch * t["epochs"]
    limit_steps = min(args.smoke_steps, total_steps) if args.smoke else total_steps
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * t["warmup_ratio"]), total_steps)
    epoch_start, batch_start, global_step, prior_seconds, prior_samples = 0, 0, 0, 0.0, 0
    if args.resume:
        # Only load training state produced by this repository, never an untrusted pickle.
        saved = torch.load(Path(args.resume) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch_start, batch_start, global_step = saved["epoch"], saved["batch"], saved["step"]
        prior_seconds, prior_samples = saved["seconds"], saved["processed_samples"]
        restore_rng(saved["rng_by_rank"][rank])
        if global_step >= limit_steps:
            raise ValueError("Checkpoint has already reached the requested step/epoch budget")
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True) if world > 1 else model
    probe_initial = probe.detach().clone()
    model.train()
    if rank == 0:
        write_json(output / "environment.json", environment())
        write_json(output / "config.json", cfg)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    processed_samples = prior_samples
    history = []

    def checkpoint(name, epoch, batch):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            save_checkpoint(output / name, model, tokenizer,
                            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "epoch": epoch, "batch": batch, "step": global_step, "rng_by_rank": states,
                             "seconds": prior_seconds + time.perf_counter() - started, "processed_samples": processed_samples}, metadata)
        if world > 1:
            dist.barrier()

    for epoch in range(epoch_start, t["epochs"]):
        sampler.set_epoch(epoch)
        # DataLoader has a separate generator so its iterator does not perturb resumed model RNG.
        loader.generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        window_loss = torch.zeros((), device=device)
        window_samples = torch.zeros((), device=device)
        for batch_index, batch in enumerate(loader):
            if epoch == epoch_start and batch_index < batch_start:
                continue
            window_begin = batch_index // accumulation * accumulation
            window_size = min(accumulation, len(loader) - window_begin)
            synchronize = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            context = contextlib.nullcontext() if synchronize or world == 1 else wrapped.no_sync()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = wrapped(**to_device(batch, device), use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch={epoch}, batch={batch_index}, rank={rank}")
                (loss / window_size).backward()
            count = batch["input_ids"].shape[0]
            window_loss += loss.detach() * count
            window_samples += count
            if synchronize:
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], t["max_grad_norm"], error_if_nonfinite=True)
                if grad_norm.item() == 0:
                    raise RuntimeError("All adapter gradients are zero")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if world > 1:
                    dist.all_reduce(window_loss)
                    dist.all_reduce(window_samples)
                processed_samples += int(window_samples.item())
                entry = {"step": global_step, "epoch": epoch, "batch": batch_index + 1,
                         "loss": (window_loss / window_samples).item(), "learning_rate": scheduler.get_last_lr()[0]}
                history.append(entry)
                if rank == 0:
                    if global_step % t["log_steps"] == 0:
                        print(json.dumps(entry), flush=True)
                    with (output / "loss.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry) + "\n")
                window_loss.zero_()
                window_samples.zero_()
                next_epoch, next_batch = (epoch + 1, 0) if batch_index + 1 == len(loader) else (epoch, batch_index + 1)
                if global_step % t["save_steps"] == 0 and global_step < limit_steps:
                    checkpoint(f"checkpoint-{global_step}", next_epoch, next_batch)
                if global_step >= limit_steps:
                    break
        if global_step >= limit_steps:
            break
    torch.cuda.synchronize(device)
    seconds = prior_seconds + time.perf_counter() - started
    peaks = [None] * world
    local = {"rank": rank, "gpu": device.index, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
             "probe_parameter": probe_name, "probe_delta_l2": (probe.detach() - probe_initial).norm().item(),
             "probe_sum": probe.detach().sum().item(), "sampler_examples": [samples[i].id for i in list(iter(sampler))[:8]]}
    if world > 1:
        dist.all_gather_object(peaks, local)
    else:
        peaks[0] = local
    if any(p["probe_delta_l2"] == 0 for p in peaks) or len({p["probe_sum"] for p in peaks}) != 1:
        raise RuntimeError("Adapter update or DDP parameter synchronization check failed")
    checkpoint("final", next_epoch, next_batch)
    if rank == 0:
        write_json(output / "precision.json", precision_report(model, optimizer))
        write_json(output / "training_metrics.json", {"experiment": cfg["experiment"], "smoke": args.smoke,
                   "training_protocol_hash": digest({k: v for k, v in metadata.items() if k != "experiment"}),
                   "steps": global_step, "seconds": seconds, "processed_samples": processed_samples,
                   "images_per_second": processed_samples / seconds, "unique_train_samples": len(samples),
                   "sampler_padding_per_epoch": len(sampler) * world - len(samples), "world_size": world,
                   "effective_batch_size": t["effective_batch_size"], "peaks_by_rank": peaks,
                   "adapter_bytes": tree_bytes(output / "final", weights_only=True),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "optimizer": "torch.optim.AdamW", "optimizer_defaults": optimizer.defaults,
                   "checkpoint_bytes": tree_bytes(output / "final"), "checkpoint_time_excluded_from_final_timing": True,
                   "history_this_invocation": history})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
