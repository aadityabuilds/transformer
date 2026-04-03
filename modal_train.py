"""
Launch transformer training on Modal with an H100 GPU.

Setup (one-time):
    pip install modal
    modal setup          # authenticates with your Modal account

    # Set your wandb API key as a Modal secret:
    modal secret create wandb-secret WANDB_API_KEY=<your-key>

Run:
    modal run modal_train.py

    # Or with custom args:
    modal run modal_train.py --max-steps 20000 --d-model 512 --num-layers 8
"""

import modal

app = modal.App("transformer-training")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6.0",
        "numpy",
        "einops>=0.8.1",
        "regex>=2024.11.6",
        "tqdm>=4.67.1",
        "wandb>=0.19.7",
    )
)

volume = modal.Volume.from_name("transformer-data", create_if_missing=True)
VOLUME_PATH = "/data"

repo_mount = modal.Mount.from_local_dir(".", remote_path="/root/transformer", condition=lambda p: not any(
    x in p for x in [".venv", "__pycache__", "wandb", "data/", ".git", "checkpoints"]
))


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    mounts=[repo_mount],
)
def train(
    train_data: str = "tinystories_valid.bin",
    val_data: str | None = None,
    num_layers: int = 4,
    d_model: int = 128,
    num_heads: int = 4,
    context_length: int = 256,
    vocab_size: int = 10000,
    batch_size: int = 32,
    max_steps: int = 10000,
    lr_max: float = 6e-4,
    lr_min: float = 6e-5,
    warmup_steps: int = 200,
    cosine_cycle_steps: int = 10000,
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

    checkpoint_dir = os.path.join(VOLUME_PATH, "checkpoints")

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


@app.function(image=image, volumes={VOLUME_PATH: volume})
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
    train_data: str = "tinystories_valid.bin",
    val_data: str = "",
    num_layers: int = 4,
    d_model: int = 128,
    num_heads: int = 4,
    context_length: int = 256,
    vocab_size: int = 10000,
    batch_size: int = 32,
    max_steps: int = 10000,
    lr_max: float = 6e-4,
    lr_min: float = 6e-5,
    warmup_steps: int = 200,
    cosine_cycle_steps: int = 10000,
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
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or None,
    )
