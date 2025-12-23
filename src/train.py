"""Training Script for MeanFlow Latent Space Edge-to-Image.

This is the main training script using pre-computed VAE latents for fast training.

Prerequisites:
    1. Preprocess dataset to latent format:
       python -m src.tools.dataset_preprocess \
           --input data/afhq_v2_raw \
           --output data/afhq_v2_processed

    2. Run training:
       python -m src.train --name my_experiment
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CONFIG, get_vae_dtype, save_config, setup_run_dir
from .datasets.latent_dataset import LatentEdgeDataset
from .models.unet import UNet
from .models.vae import VAEWrapper
from .utils.meanflow import compute_meanflow_loss, sample_multi_step, sample_one_step
from .utils.training import CSVLogger, EMAModel, Logger, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MeanFlow Edge-to-Image")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Preprocessed data directory (contains latent, edge, images)",
    )
    return parser.parse_args()


def create_dataloaders(cfg, data_dir=None):
    """Create train and validation dataloaders."""
    if data_dir:
        data_root = Path(data_dir)
    else:
        data_root = cfg["paths"]["data_root"]

    print(f"Data directory: {data_root}")

    # Check for .pt files directly in the split folders
    has_train = (data_root / "train").exists()

    if not has_train:
        raise ValueError(f"Train directory not found at {data_root / 'train'}")

    print("Initializing Datasets...")

    train_dataset = LatentEdgeDataset(
        root_dir=str(data_root),
        split="train",
        categories=cfg["data"]["categories"],
        augment=True,
    )

    val_dataset = LatentEdgeDataset(
        root_dir=str(data_root),
        split="val",
        categories=cfg["data"]["categories"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["data"]["num_workers"] > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["data"]["num_workers"] > 0),
    )

    return train_loader, val_loader


def create_model(cfg, device):
    """Create the U-Net model (edge encoder is built-in)."""

    model = UNet(
        in_channels=cfg["model"]["latent_channels"],
        edge_channels=cfg["model"]["edge_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        attention_levels=tuple(cfg["model"]["attention_levels"]),
        dropout=float(cfg["model"]["dropout"]),
    )

    # Print model info
    unet_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {unet_params / 1e6:.2f}M")

    return model.to(device)


def create_scheduler(optimizer, cfg, steps_per_epoch):
    """Creates a learning rate scheduler with warmup followed by cosine decay.

    The scheduler performs a linear warmup for a specified number of epochs,
    then switches to a cosine annealing schedule that decays the learning rate
    to a minimum value (1e-6) by the end of training.
    """
    total_epochs = cfg["training"]["total_epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    # Calculate remaining steps for the cosine decay phase
    decay_steps = max(total_steps - warmup_steps, 1)

    # Linear warmup: increases LR from 10% to 100% of base LR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    # Cosine annealing: decays LR from 100% down to 1e-6
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    ema_model,
    cfg,
    global_step,
):
    """Trains the model for one epoch.

    Notes:
        - The edge encoder is built into UNet. We pass raw edge maps to MeanFlow loss.
        - EMA tracks the main model only.
    """
    model.train()

    epoch_loss = 0.0
    epoch_logs: dict[str, float] = {}
    device = next(model.parameters()).device
    accum_steps = int(cfg["training"]["accum_steps"])

    pbar = tqdm(train_loader, leave=False, desc="Training")
    for step, (latents, edge) in enumerate(pbar):
        latents = latents.to(device, non_blocking=True).float()
        edge = edge.to(device, non_blocking=True).float()

        # Forward + backward
        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, logs = compute_meanflow_loss(model, latents, edge, cfg)

            scaler.scale(loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                global_step += 1
        else:
            loss, logs = compute_meanflow_loss(model, latents, edge, cfg)
            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                global_step += 1

        # EMA update (only when we actually stepped the optimizer)
        if (step + 1) % accum_steps == 0 and ema_model is not None:
            ema_model.update(model)

        # Accumulate metrics for logging
        epoch_loss += float(logs.get("loss", 0.0))
        for k, v in logs.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                epoch_logs[k] = epoch_logs.get(k, 0.0) + float(v)
            elif torch.is_tensor(v):
                epoch_logs[k] = epoch_logs.get(k, 0.0) + float(v.item())

        # Progress bar
        current_mse = logs.get("raw_mse", 0.0)
        pbar.set_postfix(
            {
                "loss": f"{logs.get('loss', 0.0):.4f}",
                "mse": f"{current_mse:.2e}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    # Compute epoch averages
    num_batches = max(len(train_loader), 1)
    metrics = {
        "train_loss": epoch_loss / num_batches,
        "lr": optimizer.param_groups[0]["lr"],
    }
    for key, val in epoch_logs.items():
        metrics[key] = val / num_batches

    return metrics, global_step


@torch.no_grad()
def validate(
    model,
    vae,
    val_loader,
    device,
    cfg,
    run_dir,
    epoch,
    suffix: str = "main",
    num_steps: int = 1,
):
    """Validate and save sample images.

    The model expects raw edge maps (shape [B,1,H,W]) and will encode them internally.
    """
    model.eval()

    num_samples = min(cfg["validation"]["num_samples"], len(val_loader.dataset))
    images_dir = Path(run_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_preds = []
    all_gts = []
    sample_count = 0

    for gt_latent, edge in val_loader:
        if sample_count >= num_samples:
            break

        gt_latent = gt_latent.to(device).float()
        edge = edge.to(device).float()

        # Process samples in this batch
        batch_n = gt_latent.shape[0]
        for i in range(batch_n):
            if sample_count >= num_samples:
                break

            gt_lat = gt_latent[i : i + 1]
            edge_i = edge[i : i + 1]

            # Generate latent
            if num_steps <= 1:
                pred_latent = sample_one_step(model, edge_i, latent_shape=gt_lat.shape)
            else:
                pred_latent = sample_multi_step(model, edge_i, num_steps=num_steps, latent_shape=gt_lat.shape)

            # Decode to images (optional)
            if vae is not None:
                gt_img = vae.decode(gt_lat)
                pred_img = vae.decode(pred_latent)
            else:
                # If VAE is disabled, just visualize latents (normalized)
                gt_img = gt_lat.repeat(1, 3, 1, 1)
                pred_img = pred_latent.repeat(1, 3, 1, 1)

            # Edge visualization (upsample to 256 if needed)
            if edge_i.shape[-1] != 256:
                edge_vis = F.interpolate(edge_i.repeat(1, 3, 1, 1), size=(256, 256), mode="nearest")
            else:
                edge_vis = edge_i.repeat(1, 3, 1, 1)

            # Denormalize to [0,1] for saving
            gt_img = (gt_img.clamp(-1, 1) + 1) / 2
            pred_img = (pred_img.clamp(-1, 1) + 1) / 2
            edge_vis = (edge_vis.clamp(-1, 1) + 1) / 2

            all_edges.append(edge_vis.cpu())
            all_preds.append(pred_img.cpu())
            all_gts.append(gt_img.cpu())
            sample_count += 1

    # Save grid: each row is [edge | pred | gt]
    if sample_count > 0:
        edges = torch.cat(all_edges, dim=0)
        preds = torch.cat(all_preds, dim=0)
        gts = torch.cat(all_gts, dim=0)

        rows = []
        for i in range(edges.shape[0]):
            row = torch.cat([edges[i], preds[i], gts[i]], dim=2)
            rows.append(row)

        grid = torch.stack(rows, dim=0)

        from torchvision.utils import save_image

        step_tag = f"s{int(num_steps)}"
        save_path = images_dir / f"epoch_{epoch:04d}_{suffix}_{step_tag}.png"
        save_image(grid, save_path, nrow=1, padding=2)

    model.train()
    return {"num_samples": sample_count}


def main():
    """Main training routine (aligned to new UNet; keeps resume/scheduler design)."""

    args = parse_args()
    cfg = CONFIG

    set_seed(cfg["project"]["seed"])
    device = torch.device(cfg["project"]["device"])
    print(f"Device: {device}")

    # ---------------------------------------------------------------------
    # Setup directories
    # ---------------------------------------------------------------------
    if args.resume:
        run_dir = Path(args.resume).parent.parent
    else:
        run_name = args.name or cfg["project"]["name"]
        run_dir = setup_run_dir(run_name)
        save_config(run_dir)

    logger = Logger(run_dir / "logs" / "training.log")
    csv_logger = CSVLogger(run_dir / "logs" / "metrics.csv")

    logger.log(f"\n{'=' * 60}")
    logger.log("MeanFlow Latent Space Training")
    logger.log(f"Run: {run_dir}")
    logger.log(f"{'=' * 60}")

    # ---------------------------------------------------------------------
    # Data & Model
    # ---------------------------------------------------------------------
    train_loader, val_loader = create_dataloaders(cfg, args.data_dir)

    # New: model includes edge encoder internally, so create_model returns model only
    model = create_model(cfg, device)

    # Optional VAE for visualization
    vae = None
    if cfg["validation"].get("decode_samples", True):
        logger.log("Loading VAE for visualization...")
        vae = VAEWrapper(
            model_name=cfg["vae"]["model_name"],
            device=device,
            dtype=get_vae_dtype(),
        )

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=tuple(cfg["training"]["betas"]),
    )

    # EMA Setup (model only)
    ema_model = None
    if cfg["training"]["use_ema"]:
        ema_model = EMAModel(model, cfg["training"]["ema_decay"])
        logger.log(f"EMA enabled: decay={cfg['training']['ema_decay']}")

    scaler = torch.amp.GradScaler("cuda") if cfg["training"]["use_amp"] else None

    # ---------------------------------------------------------------------
    # Resume or Start (keep your original ordering)
    # ---------------------------------------------------------------------
    start_epoch = 1
    best_metric = float("inf")  # Tracks the best Raw MSE
    global_step = 0

    if args.resume:
        # Note: scheduler is intentionally None here (you re-initialize later).
        start_epoch, best_metric, global_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler=None,
            ema_model=ema_model,
            device=device,
        )

        # Reset LR to match new schedule (keep behavior)
        base_lr = cfg["training"]["lr"]
        print(f"Force resetting Optimizer LR to {base_lr:.2e} for new schedule...")
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr
            param_group["initial_lr"] = base_lr

    # ---------------------------------------------------------------------
    # Scheduler (created after resume, as you designed)
    # IMPORTANT: scheduler.step() is called per optimizer update in train_one_epoch.
    # ---------------------------------------------------------------------
    accum_steps = int(cfg["training"]["accum_steps"])
    optimizer_steps_per_epoch = max((len(train_loader) + accum_steps - 1) // accum_steps, 1)
    scheduler = create_scheduler(optimizer, cfg, optimizer_steps_per_epoch)

    # Optional: if you want LR continuity without loading scheduler state,
    # you can fast-forward by global_step. Default off to preserve old behavior.
    if bool(cfg["training"].get("resume_fast_forward_scheduler", False)) and args.resume:
        for _ in range(int(global_step)):
            scheduler.step()

    logger.log(f"\nTraining: epoch {start_epoch} -> {cfg['training']['total_epochs']}")

    # ---------------------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------------------
    for epoch in range(start_epoch, cfg["training"]["total_epochs"] + 1):
        metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_model=ema_model,
            cfg=cfg,
            global_step=global_step,
        )

        logger.log(
            f"Epoch {epoch:3d} | "
            f"Loss: {metrics['train_loss']:.4f} | "
            f"MSE: {metrics.get('raw_mse', 0):.2e} | "
            f"LR: {metrics['lr']:.2e}"
        )

        # -----------------------------------------------------------------
        # Validation & Saving
        # -----------------------------------------------------------------
        if epoch % cfg["validation"]["interval"] == 0:
            logger.log("Running validation (generating images)...")
            steps_list = cfg["validation"].get("num_steps_list", [1, 2, 4])

            # 1) Main Model Validation
            for ns in steps_list:
                validate(
                    model=model,
                    vae=vae,
                    val_loader=val_loader,
                    device=device,
                    cfg=cfg,
                    run_dir=run_dir,
                    epoch=epoch,
                    suffix="main",
                    num_steps=int(ns),
                )

            # 2) EMA Model Validation
            if ema_model is not None:
                ema_model.apply_shadow(model)
                for ns in steps_list:
                    validate(
                        model=model,
                        vae=vae,
                        val_loader=val_loader,
                        device=device,
                        cfg=cfg,
                        run_dir=run_dir,
                        epoch=epoch,
                        suffix="ema",
                        num_steps=int(ns),
                    )
                ema_model.restore(model)

            # Save best model based on Raw MSE (preferred over weighted loss)
            current_metric = metrics.get("raw_mse", float("inf"))
            if current_metric < best_metric:
                best_metric = current_metric
                save_checkpoint(
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_loss=best_metric,
                    global_step=global_step,
                    run_dir=run_dir,
                    name="best",
                )
                logger.log(f"  ★ Best MSE: {best_metric:.2e}")

        # Logging
        csv_logger.log(epoch, metrics)

        # Regular checkpoint
        if epoch % cfg["logging"]["save_interval"] == 0:
            save_checkpoint(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_metric,
                global_step=global_step,
                run_dir=run_dir,
                name=f"epoch_{epoch:04d}",
            )

        # Latest checkpoint
        save_checkpoint(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_loss=best_metric,
            global_step=global_step,
            run_dir=run_dir,
            name="latest",
        )

    logger.log("Training complete!")


if __name__ == "__main__":
    main()
