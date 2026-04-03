import argparse
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from transformer import Transformer
from transformer.utils import (
    cross_entropy_loss,
    AdamW,
    learning_rate_schedule,
    gradient_clipping,
    data_loading,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer language model")

    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--use_rope", action="store_true", default=True)

    p.add_argument("--lr_max", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=6e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--cosine_cycle_steps", type=int, default=10000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--train_data", type=str, required=True,
                   help="Path to training data as a flat np.uint16 binary file")
    p.add_argument("--val_data", type=str, default=None,
                   help="Path to validation data (same format)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a checkpoint .pt file to resume from")

    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="transformer-lm")
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--device", type=str, default=None,
                   help="Device: cpu, mps, or cuda (auto-detected if omitted)")

    return p.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    device = get_device(args.device)
    print(f"device: {device}")

    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = (np.memmap(args.val_data, dtype=np.uint16, mode="r") if args.val_data else None)

    print(f"train tokens: {len(train_data):,}")
    if val_data is not None:
        print(f"val   tokens: {len(val_data):,}")

    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_embeddings=args.vocab_size,
        embedding_dim=args.d_model,
        theta=args.theta,
        max_seq_len=args.context_length,
        use_rope=args.use_rope,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer, device=device)
        print(f"resumed from step {start_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    model.train()
    tokens_per_step = args.batch_size * args.context_length

    pbar = tqdm(
        range(start_step, args.max_steps),
        initial=start_step,
        total=args.max_steps,
        desc="Training",
        unit="step",
    )

    for step in pbar:
        step_start = time.time()

        lr = learning_rate_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.cosine_cycle_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = data_loading(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        step_time_ms = (time.time() - step_start) * 1000
        tok_per_sec = tokens_per_step / (step_time_ms / 1000) if step_time_ms > 0 else 0.0
        loss_val = loss.item()
        perplexity = math.exp(loss_val) if loss_val < 20 else float("inf")

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            ppl=f"{perplexity:.1f}",
            lr=f"{lr:.2e}",
            toks=f"{tok_per_sec:,.0f}/s",
        )

        if step % args.log_interval == 0 and wandb_run:
            metrics = {
                "train/loss": loss_val,
                "train/perplexity": perplexity,
                "train/grad_norm": grad_norm,
                "train/step_time_ms": step_time_ms,
                "lr": lr,
                "perf/tok_per_sec": tok_per_sec,
            }
            if device.type == "cuda":
                metrics["perf/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                metrics["perf/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            elif device.type == "mps":
                metrics["perf/mps_memory_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
            wandb_run.log(metrics, step=step)

        if step > 0 and step % args.eval_interval == 0:
            losses = evaluate(
                model, train_data, val_data,
                args.batch_size, args.context_length, device, args.eval_batches,
            )
            parts = [f"eval"]
            eval_metrics = {}
            for split, val in losses.items():
                ppl = math.exp(val) if val < 20 else float("inf")
                parts.append(f"{split}_loss={val:.4f} ppl={ppl:.1f}")
                eval_metrics[f"eval/{split}_loss"] = val
                eval_metrics[f"eval/{split}_perplexity"] = ppl
            tqdm.write(f"step {step:>6d} | {' | '.join(parts)}")
            if wandb_run:
                wandb_run.log(eval_metrics, step=step)

        if step > 0 and step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            tqdm.write(f"checkpoint saved: {ckpt_path}")

    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.max_steps, final_path)
    print(f"\ntraining complete. final checkpoint: {final_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
