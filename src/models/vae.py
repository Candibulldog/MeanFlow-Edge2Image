"""VAE Wrapper for MeanFlow Latent Space Training.

This module provides a wrapper around the Stable Diffusion VAE for:
- Encoding images to latent space
- Decoding latents back to images
- Proper scaling (the 0.18215 factor)

The VAE is frozen and only used for preprocessing/postprocessing.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """Wrapper for Stable Diffusion VAE.

    This class handles:
    1. Loading the pretrained VAE
    2. Encoding images to latent space with proper scaling
    3. Decoding latents back to images
    4. Memory-efficient processing (no gradients through VAE)

    The VAE uses a scaling factor of 0.18215 to normalize latents
    to roughly unit variance, which helps training stability.

    Args:
        model_name: HuggingFace model name for the VAE.
        device: Target device for the VAE.
        dtype: Data type (float16 for memory efficiency, float32 for accuracy).
    """

    # Standard SD VAE scaling factor
    SCALING_FACTOR = 0.18215

    def __init__(
        self,
        model_name: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        print(f"Loading VAE from {model_name}...")
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

        # Freeze VAE - we never train it
        self.vae.eval()
        self.vae.requires_grad_(False)

        print("VAE loaded successfully!")
        print(f"  - Latent channels: {self.vae.config.latent_channels}")
        print(f"  - Scaling factor: {self.SCALING_FACTOR}")

    @property
    def latent_channels(self) -> int:
        """Number of latent channels (typically 4)."""
        return self.vae.config.latent_channels

    @property
    def downsample_factor(self) -> int:
        """Spatial downsampling factor (typically 8)."""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            images: RGB images [B, 3, H, W] in range [-1, 1].
            sample: If True, sample from the latent distribution.
                   If False, use the mode (mean).

        Returns:
            Latent representation [B, 4, H/8, W/8], scaled.
        """
        # Ensure correct dtype
        images = images.to(dtype=self.dtype, device=self.device)

        # Encode to latent distribution
        latent_dist = self.vae.encode(images).latent_dist

        # Sample or use mode
        if sample:
            latents = latent_dist.sample()
        else:
            latents = latent_dist.mode()

        # Apply scaling factor
        latents = latents * self.SCALING_FACTOR

        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to images.

        Args:
            latents: Latent representation [B, 4, H/8, W/8], scaled.

        Returns:
            RGB images [B, 3, H, W] in range [-1, 1].
        """
        # Ensure correct dtype
        latents = latents.to(dtype=self.dtype, device=self.device)

        # Undo scaling factor
        latents = latents / self.SCALING_FACTOR

        # Decode
        images = self.vae.decode(latents).sample

        return images

    def get_latent_size(self, image_size: int) -> int:
        """Calculate latent spatial size for given image size."""
        return image_size // self.downsample_factor
