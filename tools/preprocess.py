# tools/dataset_preprocess.py

"""
One-stop preprocessing for MeanFlow Edge-to-Image training.
Optimized for Hybrid Training Strategy:
1. Targets: Pre-encoded VAE Latents (32x32) for efficiency.
2. Conditions: PiDiNet Safe Edges (256x256) for precision (Best for Animals).

Algorithm: PiDiNet (Safe Mode)
Reasoning:
1. Distinguishes semantic boundaries (ears, eyes) from texture (fur).
2. "Safe Mode" quantization removes hidden grayscale artifacts.
3. Morphological cleanup ensures continuous contours.

Output structure:
    output_dir/
    ├── train/
    │   ├── cat/
    │   │   ├── flickr_cat_000001_latent.pt     # {latent: [4,32,32], edge: [1,256,256]}
    │   │   └── ...
    └── val/
        └── ...
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CONFIG

try:
    from controlnet_aux import PidiNetDetector
except ImportError:
    print("Error: 'controlnet_aux' is required for PiDiNet.")
    print("Please install it via: pip install controlnet_aux")
    exit(1)

# Lazy import VAE
VAEWrapper = None
# Lazy load PiDiNet
PiDiNet_Model = None


def get_vae():
    """Lazy load VAE wrapper."""
    global VAEWrapper
    if VAEWrapper is None:
        from src.models.vae import VAEWrapper as _VAEWrapper

        VAEWrapper = _VAEWrapper
    return VAEWrapper


def get_pidinet():
    """Lazy load PiDiNet Detector."""
    global PiDiNet_Model
    if PiDiNet_Model is None:
        print("Loading PiDiNet model (this may take a moment on first run)...")
        PiDiNet_Model = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    return PiDiNet_Model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess AFHQ for MeanFlow (Hybrid: Latent Target + PiDiNet Edge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Path to original AFHQ dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--image_size", type=int, default=CONFIG["data"]["image_size"], help="Target image size")
    parser.add_argument("--latent_size", type=int, default=32, help="Latent spatial size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers")
    parser.add_argument("--vae_model", type=str, default=CONFIG["vae"]["model_name"], help="VAE model name")
    parser.add_argument("--skip_latent", action="store_true", help="Skip VAE encoding")
    parser.add_argument("--save_pngs", action="store_true", help="Save visualization PNGs (Slow)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    return parser.parse_args()


def extract_pidinet_edge(image_rgb: np.ndarray, quantization_levels=8) -> np.ndarray:
    """
    Extracts edges using PiDiNet with 'Safe Mode' protocols optimized for animal faces.

    Steps:
    1. PiDiNet Inference: Detects semantic edges, suppressing fur texture.
    2. Quantization (8 levels): Removes hidden artifacts/patterns.
    3. Morphology: Closes gaps and removes noise using elliptical kernel.

    Args:
        image_rgb: [H, W, 3] numpy array, 0-255 uint8.
        quantization_levels: Number of discrete edge levels (recommend 8).

    Returns:
        edge_map: [H, W] float32 array, normalized to [0.0, 1.0].
    """
    model = get_pidinet()

    # 1. PiDiNet Inference
    # safe=True in library does some quantization, but we enforce specific levels below manually
    # to match the strict recommendation.
    # The detector expects uint8 image. Output is uint8 [0, 255].
    edges_uint8 = model(image_rgb, detect_resolution=256, safe=True)

    # Convert to numpy if it's not already (depending on library version output)
    if not isinstance(edges_uint8, np.ndarray):
        edges_uint8 = np.array(edges_uint8)

    # Ensure resize to target (just in case)
    if edges_uint8.shape[:2] != (256, 256):
        edges_uint8 = cv2.resize(edges_uint8, (256, 256))

    # 2. Explicit Safe Mode Quantization (8 levels)
    # The note recommends 8 levels over 4.
    # Formula: (val // step) * step
    step = 256 // quantization_levels
    edges_quant = (edges_uint8 // step) * step

    # 3. Morphological Post-Processing
    # Use 3x3 Ellipse kernel (better for curved animal features)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Close: Connects small gaps (e.g. in eye contours)
    edges_cleaned = cv2.morphologyEx(edges_quant, cv2.MORPH_CLOSE, kernel)

    # 4. Normalize to [0.0, 1.0] float
    edge_float = edges_cleaned.astype(np.float32) / 255.0

    return edge_float


class AFHQDataset(Dataset):
    """Dataset wrapper to handle image loading and edge extraction."""

    def __init__(self, image_paths: list[Path], output_dir: Path, args):
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.args = args

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem

        try:
            # Read image (BGR)
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                return None

            # Resize image to target size (256x256)
            image_bgr_256 = cv2.resize(
                image_bgr, (self.args.image_size, self.args.image_size), interpolation=cv2.INTER_AREA
            )
            image_rgb = cv2.cvtColor(image_bgr_256, cv2.COLOR_BGR2RGB)

            # Extract Edge using PiDiNet (Safe Mode)
            # Returns [256, 256] in range [0.0, 1.0]
            edge_256 = extract_pidinet_edge(image_rgb)

            # Optional: Save visualization PNGs
            if self.args.save_pngs:
                edge_view = (edge_256 * 255).astype(np.uint8)
                cv2.imwrite(str(self.output_dir / f"{stem}.png"), image_bgr_256)
                cv2.imwrite(str(self.output_dir / f"{stem}_edge.png"), edge_view)

            # Prepare Tensors
            # Image: [3, 256, 256] in [-1, 1] for VAE encoding
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 127.5 - 1.0

            # Edge: [1, 256, 256] in [-1, 1]
            # Mapping 0.0 -> -1.0 (Black/Background)
            # Mapping 1.0 ->  1.0 (White/Edge)
            edge_tensor = torch.from_numpy(edge_256).unsqueeze(0).float() * 2.0 - 1.0

            return {"image": image_tensor, "edge": edge_tensor, "stem": stem, "save_dir": str(self.output_dir)}

        except Exception:
            # print(f"Error processing {img_path}: {e}") # Optional: reduce spam
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images = torch.stack([b["image"] for b in batch])
    edges = torch.stack([b["edge"] for b in batch])
    stems = [b["stem"] for b in batch]
    save_dirs = [b["save_dir"] for b in batch]

    return images, edges, stems, save_dirs


@torch.no_grad()
def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("MeanFlow Preprocessing: Hybrid Mode (Latent + PiDiNet Safe Edge)")
    print("=" * 60)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print("Method:      PiDiNet (Safe Mode, 8-level Quantization)")
    print(f"Workers:     {args.num_workers}")
    print("=" * 60)

    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Initialize PiDiNet once here to ensure download happens before multiprocessing
    # Note: controlnet_aux models are usually not picklable, so we re-load/access
    # via the global lazy loader inside the worker process or keep num_workers=0 if issues arise.
    # However, PyTorch DataLoader usually works if the model is initialized inside __getitem__
    # or if we use 'spawn' start method.
    # Simple fix: The lazy loader `get_pidinet()` inside `extract_pidinet_edge` handles
    # per-process initialization naturally.

    # Load VAE
    vae = None
    if not args.skip_latent:
        print("\nLoading VAE...")
        VAEClass = get_vae()
        dtype = torch.float16 if args.dtype == "float16" else torch.float32
        vae = VAEClass(model_name=args.vae_model, device=device, dtype=dtype)
        print("VAE loaded.")

    splits = ["train", "val"] if not (input_dir / "test").exists() else ["train", "val", "test"]
    categories = ["cat", "dog", "wild"]

    total_processed = 0

    for split in splits:
        output_split = "val" if split == "test" else split

        for category in categories:
            input_cat_dir = input_dir / split / category
            output_cat_dir = output_dir / output_split / category

            if not input_cat_dir.exists():
                continue
            output_cat_dir.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(list(input_cat_dir.glob("*.png")) + list(input_cat_dir.glob("*.jpg")))
            if not image_paths:
                continue

            print(f"[{split}/{category}] Processing {len(image_paths)} images...")

            dataset = AFHQDataset(image_paths, output_cat_dir, args)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            for batch in tqdm(dataloader):
                if batch is None:
                    continue
                images, edges, stems, save_dirs = batch

                # VAE Encoding (Target)
                if vae is not None:
                    images = images.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device, enabled=(args.dtype == "float16")):
                        latents = vae.encode(images, sample=False)
                    latents = latents.cpu()
                else:
                    latents = [None] * len(images)

                # Save
                for i, stem in enumerate(stems):
                    save_data = {
                        "latent": latents[i].clone() if latents[i] is not None else None,
                        "edge": edges[i].clone(),  # [1, 256, 256]
                    }
                    torch.save(save_data, Path(save_dirs[i]) / f"{stem}_latent.pt")
                    total_processed += 1

    print(f"\nPreprocessing Complete! Total: {total_processed}")


if __name__ == "__main__":
    main()
