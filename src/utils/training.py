"""Training utilities for MeanFlow Latent Space.

This module provides essential training utilities:
    - set_seed: Reproducibility
    - EMAModel: Exponential Moving Average for model weights
    - Logger: Text logging to file and console
    - CSVLogger: Metrics logging to CSV
    - Checkpoint functions: save/load training state
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMAModel:
    """Exponential Moving Average of model parameters.

    EMA helps stabilize training and often produces better final models.
    The shadow parameters are updated as:
        shadow = decay * shadow + (1 - decay) * current

    Args:
        model: PyTorch model to track.
        decay: EMA decay rate (typically 0.9995-0.9999).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        """Update EMA parameters with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: torch.nn.Module):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: torch.nn.Module):
        """Restore original parameters after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return self.shadow.copy()

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict.copy()


class Logger:
    """Simple text logger to file and console.

    Args:
        log_file: Path to log file.
    """

    def __init__(self, log_file: str | Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, "w") as f:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            f.write(f"Training Log - Device: {device_name}\n")
            f.write("=" * 60 + "\n")

    def log(self, message: str, print_console: bool = True):
        """Write log message to file and optionally console.

        Args:
            message: String to log.
            print_console: If True, also print to stdout.
        """
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
        if print_console:
            print(message)


class CSVLogger:
    """CSV logger for training metrics.

    Logs training metrics to a CSV file for easy analysis and plotting.

    Args:
        csv_path: Path to CSV file.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Define columns for latent space training.
        # NOTE: train_loss may be adaptively weighted and thus not always a sensitive
        # indicator. We also log a few unweighted/debug metrics to make training
        # behavior diagnosable.
        columns = [
            "epoch",
            "train_loss",
            "raw_mse",
            "u_pred_rms",
            "target_rms",
            "weight_mean",
            "weight_p50",
            "lr",
            "mean_t",
            "mean_r",
            "mean_interval",
            "ratio_r_eq_t",
            "ratio_r_neq_t",
            "mean_dt_used",
            "mean_dt_used_boundary",
        ]

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def log(self, epoch: int, metrics: dict):
        """Append metrics to CSV.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric values.
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{metrics.get('train_loss', 0):.6f}",
                    f"{metrics.get('raw_mse', 0):.6f}",
                    f"{metrics.get('u_pred_rms', 0):.6f}",
                    f"{metrics.get('target_rms', 0):.6f}",
                    f"{metrics.get('weight_mean', 0):.6f}",
                    f"{metrics.get('weight_p50', 0):.6f}",
                    f"{metrics.get('lr', 0):.8f}",
                    f"{metrics.get('mean_t', 0):.4f}",
                    f"{metrics.get('mean_r', 0):.4f}",
                    f"{metrics.get('mean_interval', 0):.4f}",
                    f"{metrics.get('ratio_r_eq_t', 0):.4f}",
                    f"{metrics.get('ratio_r_neq_t', 0):.4f}",
                    f"{metrics.get('mean_dt_used', 0):.6f}",
                    f"{metrics.get('mean_dt_used_boundary', 0):.6f}",
                ]
            )


def save_checkpoint(
    model: torch.nn.Module,
    ema_model: EMAModel | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    global_step: int,
    run_dir: str | Path,
    name: str,
):
    """Save training checkpoint (single EMA: model only).

    Backward/forward policy:
      - New checkpoints write: ema_model_state_dict (if ema_model is not None)
      - We no longer save edge_encoder or ema_edge (edge encoder is built into UNet).
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "global_step": global_step,
    }

    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()

    save_path = Path(run_dir) / "models" / f"{name}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    ema_model: EMAModel | None = None,
    device: str | torch.device = "cuda",
):
    """Load checkpoint and resume training (single EMA: model only).

    Backward compatibility:
      - If "ema_model_state_dict" is missing but "ema_state_dict" exists,
        load it into ema_model (legacy single-EMA checkpoints).
      - Any legacy keys (edge_encoder_state_dict / ema_edge_state_dict) are ignored.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded model weights from epoch {checkpoint['epoch']}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✓ Loaded optimizer state")

    # Optional scheduler restore (keep your previous behavior if you want it off)
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("✓ Loaded scheduler state")

    if ema_model is not None:
        if "ema_model_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            print("✓ Loaded EMA(model) state")
        elif "ema_state_dict" in checkpoint:
            # Legacy key support
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
            print("✓ Loaded EMA(model) state (legacy key: ema_state_dict)")

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    global_step = checkpoint.get("global_step", 0)

    print(f"Resuming from epoch {start_epoch} (best loss: {best_loss:.4f})")
    return start_epoch, best_loss, global_step


def load_model_for_inference(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
):
    """Load model for inference only (single EMA: model only).

    If use_ema=True:
      - Load from ema_model_state_dict (or legacy ema_state_dict)
    else:
      - Load from model_state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if use_ema:
        if "ema_model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_model_state_dict"])
            print(f"✓ Loaded EMA(model) weights from epoch {checkpoint['epoch']}")
        elif "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
            print(f"✓ Loaded EMA(model) weights from epoch {checkpoint['epoch']} (legacy key)")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded model weights from epoch {checkpoint['epoch']} (no EMA found)")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded model weights from epoch {checkpoint['epoch']}")

    model.eval()
    return model
