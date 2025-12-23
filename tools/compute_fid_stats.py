"""
Compute FID statistics (mu, sigma) for AFHQ subsets with Automatic Resizing.

Features:
1. Independent: Does not rely on project Dataset classes (avoids Latent/Pixel mismatch).
2. Correctness: Automatically resizes Raw Images (512px) to Target Size (256px) using High-Quality Downsampling.
3. Performance: Uses Multiprocessing for fast image processing.

Usage:
  python compute_fid_stats.py --input data/afhq_v2_raw --split val --categories cat
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_fid.fid_score import InceptionV3, calculate_activation_statistics
from tqdm import tqdm

from src.config import CONFIG, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID stats with Auto-Resize.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(CONFIG["paths"]["data_root"]).replace("_processed", "_raw"),
        help="Path to ORIGINAL raw dataset (e.g., data/afhq_v2_raw).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split (usually 'val' for FID).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        choices=["cat", "dog", "wild", "all"],
        help="Categories to include.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Target size for FID calculation (MUST match generator output).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="InceptionV3 batch size.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        help="Inception feature dimensions.",
    )
    parser.add_argument(
        "--force_resize",
        action="store_true",
        help="Force regeneration of resized reference images.",
    )
    return parser.parse_args()


def process_and_save_image(args):
    """Worker function for multiprocessing."""
    src_path, dst_path, target_size = args

    if dst_path.exists():
        return

    # Read
    img = cv2.imread(str(src_path))
    if img is None:
        return

    # Resize (High Quality for Downsampling)
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Save
    cv2.imwrite(str(dst_path), img)


def prepare_reference_images(
    raw_root: Path,
    ref_root: Path,
    split: str,
    categories: list[str],
    target_size: int,
    force: bool = False,
) -> Path:
    """Resizes raw images to target size and saves them to a reference folder."""

    # Define reference directory (e.g., data/afhq_v2_256_ref/val/cat)
    # We combine all selected categories into one flat folder or keep structure?
    # pytorch-fid can handle nested folders, but usually we compute stats per subset.
    # Here we will create a specific folder for the requested subset to avoid ambiguity.

    subset_name = f"{split}_{'_'.join(categories)}"
    dest_dir = ref_root / subset_name

    if dest_dir.exists() and not force:
        # Check if empty
        if len(list(dest_dir.glob("*"))) > 10:
            print(f"Reference images found in {dest_dir}. Skipping resize.")
            return dest_dir

    if force and dest_dir.exists():
        import shutil

        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Collect all source images
    tasks = []
    print(f"Scanning raw images in {raw_root}...")

    for cat in categories:
        src_dir = raw_root / split / cat
        if not src_dir.exists():
            print(f"Warning: Source directory {src_dir} not found.")
            continue

        files = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
        for f in files:
            # Flat structure in destination to make pytorch-fid happy and simple
            dst_file = dest_dir / f"{cat}_{f.name}"
            tasks.append((f, dst_file, target_size))

    print(f"Processing {len(tasks)} images (Resize to {target_size}x{target_size})...")

    # Multiprocessing
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(tqdm(executor.map(process_and_save_image, tasks), total=len(tasks), desc="Resizing"))

    return dest_dir


def main():
    args = parse_args()
    device = torch.device(CONFIG["project"]["device"])

    # Resolve paths
    raw_root = Path(args.input)
    # create a reference folder alongside raw data
    ref_root = raw_root.parent / f"{raw_root.name.replace('_raw', '')}_{args.image_size}_ref"
    stats_root = PROJECT_ROOT / "fid_stats"
    stats_root.mkdir(exist_ok=True)

    # Resolve categories
    if args.categories == "all":
        categories = ["cat", "dog", "wild"]
        tag = "all"
    else:
        categories = [args.categories]
        tag = args.categories

    print(f"Input Raw:  {raw_root}")
    print(f"Reference:  {ref_root}")
    print(f"Target Size: {args.image_size}x{args.image_size}")

    # 1. Prepare Reference Images (Resize Real Images)
    print("\n>>> Step 1: Preparing Reference Images...")
    ref_dir = prepare_reference_images(raw_root, ref_root, args.split, categories, args.image_size, args.force_resize)

    # 2. Compute Statistics
    print(f"\n>>> Step 2: Computing Inception Statistics on {ref_dir}...")

    # Get file list
    files = sorted(list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")))
    print(f"Found {len(files)} images.")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)

    mu, sigma = calculate_activation_statistics(
        files, model, batch_size=args.batch_size, dims=args.dims, device=device, num_workers=8
    )

    # 3. Save
    out_path = stats_root / f"afhq_{tag}_{args.split}_stats.npz"
    np.savez(out_path, mu=mu, sigma=sigma)
    print(f"\n✅ FID Stats saved to: {out_path}")
    print("You can now use this .npz file to compute FID for your generated images.")


if __name__ == "__main__":
    main()
