"""Configuration for MeanFlow Latent Space Edge-to-Image Generation.

This config is designed for VAE latent space training:
- Image size: 256x256 (input) -> 32x32x4 (latent)
- Model operates on latent space for faster, higher quality training
- Uses pre-trained Stable Diffusion VAE
"""

import json
from datetime import datetime
from pathlib import Path

import torch

# Project paths
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent


def get_run_name():
    """Generates unique run name with timestamp.

    Returns:
        str: Format 'YYYYMMDD_HHMM'
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


CONFIG = {
    # ===========================================
    # Project
    # ===========================================
    "project": {
        "name": "MeanFlow_Edge2Img",
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    # ===========================================
    # Paths
    # ===========================================
    "paths": {
        "data_root": PROJECT_ROOT / "data" / "afhq_v2_processed",
        "checkpoint_root": PROJECT_ROOT / "checkpoints",
    },
    # ===========================================
    # VAE
    # ===========================================
    "vae": {
        "model_name": "stabilityai/sd-vae-ft-mse",
        "dtype": "float16",
        "scaling_factor": 0.18215,
    },
    # ===========================================
    # Data
    # ===========================================
    "data": {
        "image_size": 256,
        "latent_size": 32,
        "latent_channels": 4,
        "edge_channels": 1,
        "categories": ["all"],
        "num_workers": 8,
        "edge_type": "pidinet_safe",
        "edge_dilate_iter": 1,
    },
    # ===========================================
    # Model
    # ===========================================
    "model": {
        "latent_channels": 4,
        "edge_channels": 1,
        "out_channels": 4,
        "base_channels": 128,
        "channel_mults": (1, 2, 4),
        "num_res_blocks": 2,
        "attention_levels": [2],
        "dropout": 0.1,
    },
    # ===========================================
    # Training
    # ===========================================
    "training": {
        "total_epochs": 480,
        "batch_size": 64,
        "accum_steps": 4,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.99),
        "grad_clip": 1.0,
        # LR schedule
        "warmup_epochs": 10,
        # Mixed precision
        "use_amp": True,
        # EMA
        "use_ema": True,
        "ema_decay": 0.999,
        # MeanFlow (paper optimal settings)
        "boundary_ratio": 0.25,  # r == t
        "time_dist": "logit_normal",
        "time_mean": -0.4,
        "time_std": 1.0,
        "adaptive_weight_p": 1.0,
        "adaptive_weight_c": 1e-6,
        # Condition dropout (Crucial for CFG)
        "cond_drop_prob": 0.1,  # Probability to drop condition during training
        # Paper-style CFG (training-time guidance strength omega; Eq. 13 / 19)
        "cfg_omega": 2.0,
    },
    # ===========================================
    # Validation
    # ===========================================
    "validation": {
        "interval": 10,
        "num_samples": 8,
        "save_images": True,
        "decode_samples": True,
        "num_steps_list": [1, 2, 4],
    },
    # ===========================================
    # Logging
    # ===========================================
    "logging": {
        "save_interval": 30,
    },
}


# ===========================================
# Helper Functions
# ===========================================


def setup_run_dir(run_name=None):
    """Create run directory structure."""
    if run_name is None:
        run_name = f"{CONFIG['project']['name']}_{get_run_name()}"

    run_dir = CONFIG["paths"]["checkpoint_root"] / run_name

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    return run_dir


def save_config(run_dir):
    """Save config to JSON."""
    config_dict = {}
    for section, values in CONFIG.items():
        if isinstance(values, dict):
            config_dict[section] = {k: str(v) if isinstance(v, Path) else v for k, v in values.items()}
        else:
            config_dict[section] = values

    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def get_vae_dtype():
    """Get VAE dtype from config."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[CONFIG["vae"]["dtype"]]


def print_config():
    """Print config summary."""
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)

    for section, values in CONFIG.items():
        print(f"\n[{section.upper()}]")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k:20s} = {v}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config()
