"""
Launch transformer training on Modal with an H100 GPU.

Setup (one-time):
    pip install modal
    modal setup          # authenticates with your Modal account
    modal secret create wandb-secret WANDB_API_KEY=<your-key>

Prepare TinyStories data:
    modal run modal_train.py::prepare_data

Train (detached, survives terminal disconnect):
    modal run --detach modal_train.py --wandb-run-name "77M-tinystories"

Download checkpoint after training:
    modal volume get transformer-data tinystories_checkpoints/final.pt ./checkpoints/final.pt
"""

import modal

app = modal.App("transformer-training")

IGNORED_PATHS = {".venv", "__pycache__", "wandb", "data", ".git", "checkpoints", ".cursor"}

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6.0",
        "numpy",
        "einops>=0.8.1",
        "regex>=2024.11.6",
        "tqdm>=4.67.1",
        "wandb>=0.19.7",
        "tiktoken>=0.7.0",
        "datasets>=2.19.0",
    )
)

train_image = base_image.add_local_dir(
    ".",
    remote_path="/root/transformer",
    ignore=lambda p: any(part in IGNORED_PATHS for part in p.parts),
)

volume = modal.Volume.from_name("transformer-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=base_image,
    timeout=60 * 60 * 2,
    volumes={VOLUME_PATH: volume},
)
def prepare_data(
    dataset_name: str = "roneneldan/TinyStories",
    dataset_config: str = "",
    max_tokens: int = 600_000_000,
    output_name: str = "tinystories_train.bin",
):
    """Download and tokenize a HuggingFace dataset with tiktoken GPT-2 encoding."""
    import os
    os.environ["HF_HOME"] = "/tmp/hf_home"

    import numpy as np
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    print(f"Tokenizer: gpt2 ({enc.n_vocab} tokens, EOT={eot})")
    print(f"Target: up to {max_tokens:,} tokens from {dataset_name}")

    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    output_path = os.path.join(VOLUME_PATH, output_name)
    buffer = []
    total_tokens = 0
    doc_count = 0
    FLUSH_SIZE = 10_000_000

    with open(output_path, "wb") as f:
        for example in ds:
            tokens = enc.encode(example["text"])
            tokens.append(eot)
            buffer.extend(tokens)
            doc_count += 1

            if len(buffer) >= FLUSH_SIZE:
                write_count = min(len(buffer), max_tokens - total_tokens)
                arr = np.array(buffer[:write_count], dtype=np.uint16)
                arr.tofile(f)
                total_tokens += write_count
                buffer = buffer[write_count:] if write_count < len(buffer) else []
                print(f"  {total_tokens:,} tokens | {doc_count:,} docs")

                if total_tokens >= max_tokens:
                    break

        if buffer and total_tokens < max_tokens:
            write_count = min(len(buffer), max_tokens - total_tokens)
            arr = np.array(buffer[:write_count], dtype=np.uint16)
            arr.tofile(f)
            total_tokens += write_count

    file_mb = os.path.getsize(output_path) / 1e6
    print(f"\nDone: {total_tokens:,} tokens from {doc_count:,} documents")
    print(f"Saved: {output_path} ({file_mb:.1f} MB)")

    volume.commit()
    print("Volume committed.")


@app.function(
    image=train_image,
    gpu="H100",
    timeout=60 * 60 * 2 + 60 * 15,  # hard limit: 2h15m (2h training + startup buffer)
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    train_data: str = "tinystories_train.bin",
    val_data: str | None = None,
    num_layers: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    context_length: int = 512,
    vocab_size: int = 50257,
    batch_size: int = 64,
    max_steps: int = 19000,
    lr_max: float = 6e-4,
    lr_min: float = 6e-5,
    warmup_steps: int = 1000,
    cosine_cycle_steps: int = 19000,
    checkpoint_subdir: str = "tinystories_checkpoints",
    wandb_project: str = "transformer-lm",
    wandb_run_name: str | None = None,
):
    import subprocess
    import sys
    import os

    repo_dir = "/root/transformer"
    os.chdir(repo_dir)

    train_path = os.path.join(VOLUME_PATH, train_data)
    val_args = []
    if val_data:
        val_path = os.path.join(VOLUME_PATH, val_data)
        val_args = ["--val_data", val_path]

    checkpoint_dir = os.path.join(VOLUME_PATH, checkpoint_subdir)

    cmd = [
        sys.executable, "transformer/train.py",
        "--train_data", train_path,
        *val_args,
        "--num_layers", str(num_layers),
        "--d_model", str(d_model),
        "--num_heads", str(num_heads),
        "--context_length", str(context_length),
        "--vocab_size", str(vocab_size),
        "--batch_size", str(batch_size),
        "--max_steps", str(max_steps),
        "--lr_max", str(lr_max),
        "--lr_min", str(lr_min),
        "--warmup_steps", str(warmup_steps),
        "--cosine_cycle_steps", str(cosine_cycle_steps),
        "--checkpoint_dir", checkpoint_dir,
        "--wandb",
        "--wandb_project", wandb_project,
    ]
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])

    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

    volume.commit()


@app.function(image=base_image, volumes={VOLUME_PATH: volume})
def upload_data(local_path: str, remote_name: str | None = None):
    """Upload a local data file to the Modal volume."""
    import os
    import shutil

    dest_name = remote_name or os.path.basename(local_path)
    dest = os.path.join(VOLUME_PATH, dest_name)
    shutil.copy2(local_path, dest)
    volume.commit()
    print(f"Uploaded {local_path} -> {dest}")


@app.local_entrypoint()
def main(
    train_data: str = "tinystories_train.bin",
    val_data: str = "",
    num_layers: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    context_length: int = 512,
    vocab_size: int = 50257,
    batch_size: int = 64,
    max_steps: int = 19000,
    lr_max: float = 6e-4,
    lr_min: float = 6e-5,
    warmup_steps: int = 1000,
    cosine_cycle_steps: int = 19000,
    checkpoint_subdir: str = "tinystories_checkpoints",
    wandb_project: str = "transformer-lm",
    wandb_run_name: str = "",
):
    train.remote(
        train_data=train_data,
        val_data=val_data or None,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        context_length=context_length,
        vocab_size=vocab_size,
        batch_size=batch_size,
        max_steps=max_steps,
        lr_max=lr_max,
        lr_min=lr_min,
        warmup_steps=warmup_steps,
        cosine_cycle_steps=cosine_cycle_steps,
        checkpoint_subdir=checkpoint_subdir,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or None,
    )
