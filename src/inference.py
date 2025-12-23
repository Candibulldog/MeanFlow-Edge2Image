"""MeanFlow Edge-to-Image Inference Script (Clean Version).

Generates images from edge maps using a trained MeanFlow model.
Simplified to remove velocity scaling (uses raw model predictions).
Includes support for FID generation and step comparison.

Usage:
    # 1. Basic inference (Grid visualization)
    python -m src.inference --ckpt checkpoints/best.pt

    # 2. Compare steps (1 vs 2 vs 4)
    python -m src.inference --ckpt checkpoints/best.pt --compare_steps

    # 3. FID Generation (Process ALL images, save raw predictions)
    python -m src.inference --ckpt checkpoints/best.pt --fid
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import CONFIG, get_vae_dtype
from src.models.unet import UNet
from src.models.vae import VAEWrapper
from src.utils.training import load_model_for_inference


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="MeanFlow Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (train/val).")
    parser.add_argument("--category", type=str, default="all", help="Category or 'all'.")
    parser.add_argument("--num_samples", type=int, default=20, help="Samples per category (ignored if --fid is set).")
    parser.add_argument("--steps", type=int, default=1, help="Number of sampling steps.")

    # Feature flags
    parser.add_argument("--compare_steps", action="store_true", help="Compare 1/2/4 steps.")
    parser.add_argument(
        "--fid", action="store_true", help="FID Mode: Generate all validation images without grid concatenation."
    )

    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, Path]:
    """Loads model from checkpoint with EMA weights."""
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.json"

    with open(config_path) as f:
        cfg = json.load(f)

    model = UNet(
        in_channels=cfg["model"]["latent_channels"],
        edge_channels=cfg["model"]["edge_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attention_levels=tuple(cfg["model"]["attention_levels"]),
        dropout=cfg["model"]["dropout"],
    )
    model.to(device)

    # Use load_model_for_inference to properly load EMA weights
    load_model_for_inference(ckpt_path, model, device=device, use_ema=True)

    return model, cfg, run_dir


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    edge: torch.Tensor,
    steps: int = 1,
) -> torch.Tensor:
    """Generates latent using MeanFlow sampling (Standard Euler integration)."""
    device = edge.device
    B = edge.shape[0]
    z = torch.randn(B, 4, 32, 32, device=device, dtype=edge.dtype)

    if steps == 1:
        # One-step generation: z0 = z1 - v(z1, 0, 1)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        u = model(z, r, t, edge)
        return z - u

    # Multi-step generation
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_curr, t_next = timesteps[i], timesteps[i + 1]
        t_in = torch.full((B,), float(t_curr), device=device)
        r_in = torch.full((B,), float(t_next), device=device)
        u = model(z, r_in, t_in, edge)

        # Euler update
        dt = float(t_curr - t_next)
        z = z - dt * u

    return z


@torch.no_grad()
def generate_comparison(
    model: torch.nn.Module,
    vae: VAEWrapper,
    edge: torch.Tensor,
    gt_latent: torch.Tensor,
    steps_list: list[int] | None = None,
) -> torch.Tensor:
    """Generates comparison grid (Edge | GT | Steps...)."""
    images = []

    # Edge visualization
    edge_vis = (edge.repeat(1, 3, 1, 1).clamp(-1, 1) + 1) / 2
    images.append(edge_vis)

    # Ground truth
    gt_img = (vae.decode(gt_latent).clamp(-1, 1) + 1) / 2
    images.append(gt_img)

    # Generate for each step count
    for st in steps_list:
        pred_latent = generate(model, edge, steps=st)
        pred = (vae.decode(pred_latent).clamp(-1, 1) + 1) / 2
        images.append(pred)

    return torch.cat(images, dim=3)


def main():
    """Main inference entry point."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)

    # Load model
    model, cfg, run_dir = load_model(ckpt_path, device)
    print(f"Loaded: {ckpt_path}")

    # Load VAE
    vae = VAEWrapper(CONFIG["vae"]["model_name"], device=device, dtype=get_vae_dtype())
    vae.eval()

    # Setup paths
    data_root = Path(cfg["paths"]["data_root"]) / args.split

    # ---------------------------------------------------------
    # Configure Output Directory & Mode
    # ---------------------------------------------------------
    if args.fid:
        print(">>> Mode: FID Generation (Processing ALL validation images)")
        # Force processing of all images for FID statistics
        args.num_samples = 99999999
        out_name = f"fid_samples_s{args.steps}"
    elif args.compare_steps:
        out_name = "steps_comparison"
    else:
        out_name = f"samples_s{args.steps}"

    output_dir = run_dir / "results" / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Get categories
    categories = [args.category] if args.category != "all" else [d.name for d in data_root.iterdir() if d.is_dir()]

    # Process
    for cat in categories:
        cat_dir = data_root / cat
        if not cat_dir.exists():
            continue

        # Only create sub-directory if NOT in FID mode
        # (pytorch-fid cannot recursively search sub-directories)
        if not args.fid:
            (output_dir / cat).mkdir(exist_ok=True)

        # Load all files if FID mode, else subset
        all_files = sorted(cat_dir.glob("*_latent.pt"))
        files = all_files if args.fid else all_files[: args.num_samples]

        for f in tqdm(files, desc=cat):
            data = torch.load(f, map_location=device, weights_only=True)

            # ---------------------------------------------------------
            # Fix: Handle PiDiNet RGB Output Mismatch
            # PiDiNet may save edges as [C, H, W] or [H, W, C] with C=3.
            # We enforce [1, H, W] grayscale here.
            # ---------------------------------------------------------
            raw_edge = data["edge"]

            # Case 1: [1, 256, 256, 3] or [4, 256, 256] -> Slice first channel
            if raw_edge.dim() == 4 and raw_edge.shape[-1] == 3:
                # Remove batch & last dim RGB
                raw_edge = raw_edge[0, ..., 0]

            # Case 2: [256, 256, 3] -> Slice last dim
            elif raw_edge.dim() == 3 and raw_edge.shape[-1] == 3:
                raw_edge = raw_edge[..., 0]

            # Case 3: [3, 256, 256] -> Slice first dim
            elif raw_edge.dim() == 3 and raw_edge.shape[0] == 3:
                raw_edge = raw_edge[0]

            # Prepare tensors: Edge must be [1, 1, 256, 256]
            if raw_edge.dim() == 2:
                raw_edge = raw_edge.unsqueeze(0)  # [1, 256, 256]

            edge = raw_edge.unsqueeze(0).to(device).float()
            gt_latent = data["latent"].unsqueeze(0).to(device).float()
            name = f.stem.replace("_latent", "") + ".png"

            # ---------------------------------------------------------
            # Generation Logic
            # ---------------------------------------------------------
            if args.fid:
                # FID Mode: Generate and save ONLY the prediction image (clean).
                # Save to FLATTENED directory (no subfolders) for pytorch-fid compatibility.
                pred = generate(model, edge, args.steps)
                pred_img = (vae.decode(pred).clamp(-1, 1) + 1) / 2

                # Add category prefix to filename to avoid collisions
                save_name = f"{cat}_{name}"
                save_image(pred_img, output_dir / save_name)

            elif args.compare_steps:
                # Comparison Mode: Save grid [Edge | GT | s1 | s2 | s4]
                grid = generate_comparison(model, vae, edge, gt_latent, steps_list=[1, 2, 4])
                save_image(grid, output_dir / cat / name)

            else:
                # Default Mode: Save grid [Edge | Pred | GT]
                pred = generate(model, edge, args.steps)
                pred_img = (vae.decode(pred).clamp(-1, 1) + 1) / 2
                gt_img = (vae.decode(gt_latent).clamp(-1, 1) + 1) / 2

                # Expand edge to 3 channels for visualization
                edge_vis = (edge.repeat(1, 3, 1, 1).clamp(-1, 1) + 1) / 2

                grid = torch.cat([edge_vis, pred_img, gt_img], dim=3)
                save_image(grid, output_dir / cat / name)

    print("Done.")

    print("Done.")


if __name__ == "__main__":
    main()
