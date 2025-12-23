"""U-Net architecture for MeanFlow with Multi-scale Edge Injection.

This implementation follows the simplified U-Net structure similar to DDPM/EDM,
adapted for latent space generation with edge conditioning.

Key design choices:
1. ChannelLayerNorm instead of GroupNorm for JVP stability
2. Zero-initialization on output layer
3. Multi-scale edge injection at each resolution level
4. Dual time embedding (t, t-r) as per MeanFlow paper Table 1c
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """Layer normalization over channel dimension.

    More memory-efficient than GroupNorm for JVP computation.
    Computes statistics across channel dimension for each spatial position.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time conditioning."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = ChannelLayerNorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = ChannelLayerNorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),
        )
        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time embedding injection (scale and shift)
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        scale, shift = time_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Attention(nn.Module):
    """Self-attention layer for capturing long-range dependencies."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = ChannelLayerNorm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)

        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bhnm,bhcm->bhcn", attn, v)

        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class Downsample(nn.Module):
    """Downsampling layer using strided convolution (can change channels)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer using nearest neighbor + convolution."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class MultiScaleEdgeEncoder(nn.Module):
    """Encodes high-resolution edge maps to multi-scale features.

    This module first compresses the 256x256 pixel-space edge map into a 32x32
    latent-space feature map using a lightweight stem (Tiny Encoder). It then
    generates multi-scale features (32x32, 16x16, 8x8) to be injected at different
    levels of the U-Net.

    The output layers of each scale are initialized to zero to ensure the model
    training starts from a stable unconditional state (Zero Convolution strategy).
    """

    def __init__(self, in_channels, channel_mults, base_channels):
        super().__init__()

        # Stem: Compresses 256x256 edge input to 32x32 feature map
        # Uses 3 layers of stride-2 convolutions (256 -> 128 -> 64 -> 32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, base_channels, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
        )

        self.encoders = nn.ModuleList()

        # Build feature extractors for each scale starting from the 32x32 stem output
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            layers = []

            # Determine downsampling requirements relative to the 32x32 stem
            # Level 0 (32x32): No downsampling
            # Level 1 (16x16): 1 downsample
            # Level 2 (8x8): 2 downsamples
            num_downsamples = i

            current_ch = base_channels
            for _ in range(num_downsamples):
                layers.extend(
                    [
                        nn.Conv2d(current_ch, out_ch, kernel_size=3, stride=2, padding=1),
                        nn.SiLU(),
                    ]
                )
                current_ch = out_ch

            # Final projection layer to match U-Net channel width at this level
            layers.append(nn.Conv2d(current_ch, out_ch, kernel_size=3, padding=1))

            self.encoders.append(nn.Sequential(*layers))

        self._zero_init_outputs()

    def _zero_init_outputs(self):
        """Zero-initializes the last convolution layer of each encoder branch."""
        for encoder in self.encoders:
            # Find the last Conv2d layer in the sequential block
            last_conv = None
            for module in encoder.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module

            if last_conv is not None:
                nn.init.zeros_(last_conv.weight)
                nn.init.zeros_(last_conv.bias)

    def forward(self, edge):
        """Encodes the edge map.

        Args:
            edge: High-resolution edge map tensor [B, 1, 256, 256].

        Returns:
            List of feature tensors at different resolutions (e.g., 32x32, 16x16, 8x8).
        """
        # Fold 256x256 -> 32x32
        stem_feat = self.stem(edge)

        # Generate features for each scale
        features = []
        for encoder in self.encoders:
            features.append(encoder(stem_feat))

        return features


class UNet(nn.Module):
    """U-Net with Multi-scale Edge Injection for MeanFlow.

    Architecture:
    - Encoder: 3 levels (32→16→8) with ResBlocks
    - Bottleneck: ResBlocks + Attention at 8×8
    - Decoder: 3 levels (8→16→32) with skip connections
    - Edge injection: Add edge features at each encoder level

    Args:
        in_channels: Input latent channels (default: 4 for SD VAE)
        edge_channels: Edge condition channels (default: 1)
        out_channels: Output channels (default: 4)
        base_channels: Base channel width (default: 128)
        channel_mults: Channel multipliers per level (default: (1, 2, 4))
        num_res_blocks: ResBlocks per level (default: 2)
        attention_levels: Which levels to apply attention (default: (2,) for 8×8 only)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        in_channels=4,
        edge_channels=1,
        out_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        attention_levels=(2,),
        dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        num_levels = len(channel_mults)

        # Time embedding
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Dual time embedding for (t, t-r) as per MeanFlow paper
        self.interval_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Multi-scale edge encoder
        self.edge_encoder = MultiScaleEdgeEncoder(
            in_channels=edge_channels,
            channel_mults=channel_mults,
            base_channels=base_channels,
        )

        # Input convolution (only latent, edge is injected separately)
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        encoder_channels = []  # store ONLY skip channels that will be concatenated later

        for level in range(num_levels):
            out_ch = base_channels * channel_mults[level]

            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                encoder_channels.append(ch)  # ONLY after ResBlock

                if level in attention_levels:
                    self.encoder_blocks.append(Attention(ch))  # no encoder_channels append

            if level < num_levels - 1:
                next_ch = base_channels * channel_mults[level + 1]
                self.downsamplers.append(Downsample(ch, next_ch))
                ch = next_ch

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [
                ResBlock(ch, ch, time_dim, dropout),
                Attention(ch),
                ResBlock(ch, ch, time_dim, dropout),
            ]
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mults[level]

            for _ in range(num_res_blocks):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch

                if level in attention_levels:
                    self.decoder_blocks.append(Attention(ch))

            if level > 0:
                self.upsamplers.append(Upsample(ch))

        # Output
        self.output_norm = ChannelLayerNorm(ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Zero-initialize output for stable training start
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, r, t, edge):
        """Forward pass of the U-Net.

        Args:
            x: Noisy latent tensor [B, 4, 32, 32].
            r: Start time tensor [B].
            t: End time tensor [B].
            edge: High-resolution edge condition tensor [B, 1, 256, 256].

        Returns:
            Predicted velocity field tensor [B, 4, 32, 32].
        """
        # Compute time embeddings
        t_emb = self.time_mlp(t)
        interval_emb = self.interval_mlp(t - r)
        time_emb = t_emb + interval_emb

        # Encode edge map to multi-scale features
        edge_features = self.edge_encoder(edge)

        # Initial projection of the input latent
        h = self.input_conv(x)

        # Inject edge features at the first resolution level (32x32)
        h = h + edge_features[0]

        # Encoder pathway
        skips = []
        edge_idx = 0
        downsample_idx = 0
        block_idx = 0
        num_levels = len(self.channel_mults)

        for level in range(num_levels):
            for _ in range(self.num_res_blocks):
                block = self.encoder_blocks[block_idx]
                h = block(h, time_emb)
                block_idx += 1

                # Apply attention if present at this block index
                if block_idx < len(self.encoder_blocks) and isinstance(self.encoder_blocks[block_idx], Attention):
                    h = self.encoder_blocks[block_idx](h)
                    block_idx += 1

                skips.append(h)

            # Downsample and inject next level of edge features
            if level < num_levels - 1:
                h = self.downsamplers[downsample_idx](h)
                downsample_idx += 1

                edge_idx += 1
                if edge_idx < len(edge_features):
                    h = h + edge_features[edge_idx]

        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResBlock):
                h = block(h, time_emb)
            else:
                h = block(h)

        # Decoder pathway
        block_idx = 0
        upsample_idx = 0

        for level in reversed(range(num_levels)):
            for _ in range(self.num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)

                block = self.decoder_blocks[block_idx]
                h = block(h, time_emb)
                block_idx += 1

                if block_idx < len(self.decoder_blocks) and isinstance(self.decoder_blocks[block_idx], Attention):
                    h = self.decoder_blocks[block_idx](h)
                    block_idx += 1

            if level > 0:
                h = self.upsamplers[upsample_idx](h)
                upsample_idx += 1

        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h
