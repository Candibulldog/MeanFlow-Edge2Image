"""MeanFlow Loss Module for Latent Space Edge-to-Image Generation.

This module implements the core MeanFlow training algorithm adapted for VAE
latent space. It includes the solver for the Ordinary Differential Equation (ODE)
transport problem and supports Training-time Classifier-Free Guidance (CFG).

References:
    Geng et al., "Mean Flows for One-step Generative Modeling", arXiv 2025.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.func import jvp


def sample_time_pairs(
    batch_size: int,
    device: torch.device,
    distribution: str = "logit_normal",
    mean: float = -0.4,
    std: float = 1.0,
    boundary_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples (r, t) time pairs for MeanFlow training.

    Implements the training distribution described in Section 4.3 and Table 1a.
    The data is a mixture of "Interval Cases" (learning average velocity) and
    "Boundary Cases" (learning instantaneous velocity).

    Args:
        batch_size: Number of samples in the batch.
        device: The computing device.
        distribution: Sampling distribution ('logit_normal' or 'uniform').
        mean: Mean for logit-normal distribution (Ref: Table 1a uses -0.4).
        std: Std for logit-normal distribution (Ref: Table 1a uses 1.0).
        boundary_ratio: Probability \rho of sampling boundary cases (r=t).
                        (Ref: Table 1a optimal value is 0.25).

    Returns:
        r: Start time tensor of shape [B].
        t: End time tensor of shape [B], guaranteeing t >= r.
        boundary_mask: Boolean mask where True indicates r == t.
    """
    if distribution == "logit_normal":
        # Logit-Normal sampling as per Esser et al. (2024)
        n1 = torch.randn(batch_size, device=device) * std + mean
        n2 = torch.randn(batch_size, device=device) * std + mean
        t1 = torch.sigmoid(n1)
        t2 = torch.sigmoid(n2)
    else:
        t1 = torch.rand(batch_size, device=device)
        t2 = torch.rand(batch_size, device=device)

    # Enforce order: r is start time, t is end time.
    t = torch.maximum(t1, t2)
    r = torch.minimum(t1, t2)

    # Apply Boundary Condition sampling (r=t) with probability \rho.
    boundary_mask = torch.rand(batch_size, device=device) < boundary_ratio
    r = torch.where(boundary_mask, t, r)

    return r, t, boundary_mask


def compute_meanflow_loss(
    model: torch.nn.Module,
    x_latent: torch.Tensor,
    edge: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the MeanFlow training loss with Training-time CFG.

    This function constructs the MeanFlow target using the self-consistency
    identity derived in the paper. It uses Jacobian-Vector Products (JVP)
    to compute the total derivative of the velocity field.

    Ref: Eq. (10) for the optimization objective.

    Args:
        model: Neural network predicting velocity u_theta(z, r, t | cond).
        x_latent: Ground truth VAE latents [B, 4, H, W].
        edge: Edge condition map [B, 1, H, W].
        cfg: Configuration dictionary containing training hyperparameters.

    Returns:
        loss: Scalar loss value.
        logs: Dictionary of metrics for logging.
    """
    device = x_latent.device
    dtype = x_latent.dtype
    B = x_latent.shape[0]
    tcfg = cfg["training"]

    # -------------------------------------------------------------------------
    # 1. Time Sampling
    # -------------------------------------------------------------------------
    r, t, boundary_mask = sample_time_pairs(
        B,
        device,
        distribution=tcfg.get("time_dist", "logit_normal"),
        mean=tcfg.get("time_mean", -0.4),
        std=tcfg.get("time_std", 1.0),
        boundary_ratio=tcfg.get("boundary_ratio", 0.25),
    )
    t_b = t.view(B, 1, 1, 1)

    # -------------------------------------------------------------------------
    # 2. Construct Flow Path (Interpolation)
    # -------------------------------------------------------------------------
    # Linear interpolation between data (x_latent) and noise (eps).
    # z_t = (1-t) * x + t * eps
    eps = torch.randn_like(x_latent)
    z_t = (1.0 - t_b) * x_latent + t_b * eps

    # The true vector field connecting x_latent to eps.
    # v_t = eps - x_latent (points towards noise)
    v_t = eps - x_latent

    # -------------------------------------------------------------------------
    # 3. Training-time Guidance (CFG) Logic
    # Ref: Appendix B.1 "Training-time Guidance"
    # -------------------------------------------------------------------------
    # We construct a "Guided Velocity Field" on the fly during training.
    # This ensures the model learns the trajectory of the guided sampler directly.
    #
    # Key insight: CFG is applied to the TARGET, not the model output.
    # - Dropped samples: train against raw v_t (unconditional distribution)
    # - Non-dropped samples: train against guided velocity v_guided

    null_edge = torch.zeros_like(edge)
    omega = float(tcfg.get("cfg_omega", 1.0))
    cond_drop_prob = float(tcfg.get("cond_drop_prob", 0.1))

    # Sample-wise condition dropout
    # Each sample independently decides whether to drop its condition
    drop_mask = torch.rand(B, device=device) < cond_drop_prob
    drop_mask_expanded = drop_mask.view(B, 1, 1, 1)

    # Prepare training condition: null for dropped samples, edge otherwise
    train_edge = torch.where(drop_mask_expanded, null_edge, edge)

    # Compute target velocity with CFG for non-dropped samples
    if omega != 1.0:
        # We need u_uncond to calculate guidance for non-dropped samples
        with torch.no_grad():
            u_tt_uncond = model(z_t, t, t, edge=null_edge)

        # Construct the guided target vector
        # v_guided = omega * v_true + (1 - omega) * u_uncond
        v_guided = omega * v_t + (1.0 - omega) * u_tt_uncond

        # Apply CFG only to non-dropped samples
        # Dropped samples train against raw v_t to learn unconditional distribution
        flow_target_v = torch.where(drop_mask_expanded, v_t, v_guided).detach()
    else:
        # No CFG (omega = 1.0): all samples use raw v_t
        flow_target_v = v_t

    # -------------------------------------------------------------------------
    # 4. Total Derivative Computation via JVP
    # Ref: Eq. (8) "Total derivative with respect to t"
    # d/dt u = (grad_z u) * v + (partial_t u)
    # -------------------------------------------------------------------------
    def model_fn(z_in, r_in, t_in):
        return model(z_in, r_in, t_in, edge=train_edge)

    primals = (z_t, r, t)

    # Tangent vector corresponds to the direction of differentiation in Eq. (8):
    # - Spatial component (dz/dt): v_t (TRUE instantaneous velocity)
    # - Start time component (dr/dt): 0 (r is fixed for the integral)
    # - End time component (dt/dt): 1 (derivative w.r.t t)
    #
    # CRITICAL: The MeanFlow Identity derivation assumes dz/dt = v(z_t, t),
    # which is the TRUE conditional velocity (eps - x), NOT the CFG-modified velocity.
    # Using flow_target_v here breaks the mathematical foundation.
    tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))

    # Compute u_pred (output) and du_dt (total derivative) in one pass
    # Overhead is small as it only backprops to inputs (Ref: Appendix A)
    u_pred, du_dt = jvp(model_fn, primals, tangents)

    # -------------------------------------------------------------------------
    # 5. Construct MeanFlow Target
    # Ref: Eq. (10) "Target construction with stop-gradient"
    # u_target = v - (t - r) * (d/dt u)
    # -------------------------------------------------------------------------
    dt_raw = (t - r).view(B, 1, 1, 1)

    # Handle boundary cases where t=r (dt=0) to avoid numerical issues
    dt = torch.where(
        boundary_mask.view(B, 1, 1, 1),
        torch.zeros_like(dt_raw),
        dt_raw.clamp(min=1e-6),
    )

    # Target uses flow_target_v (which may include CFG), but JVP used true v_t
    raw_target_val = flow_target_v - dt * du_dt

    # Implementation Detail: Target Clipping
    # Constraints the target value to a physically reasonable range for VAE latents.
    # This prevents gradient explosions caused by transient spikes in JVP.
    raw_target_val = raw_target_val.clamp(-8.0, 8.0)

    # Apply Stop-Gradient (Ref: Eq. (10) "sg(...)")
    # The target is the corrected velocity field.
    target = raw_target_val.detach().to(dtype)

    # -------------------------------------------------------------------------
    # 6. Compute Loss
    # Ref: Eq. (9) "MeanFlow Objective"
    # -------------------------------------------------------------------------
    error = u_pred.to(dtype) - target

    # Adaptive Weighting (Ref: Table 1e)
    # p=1.0 is recommended for Latent Space to reduce variance.
    p = float(tcfg.get("adaptive_weight_p", 1.0))
    c_eps = float(tcfg.get("adaptive_weight_c", 1e-6))

    if p > 0.0:
        # Weight = 1 / (|error|^2 + epsilon)^p
        error_sq = (error**2).mean(dim=[1, 2, 3], keepdim=True)
        weight = (1.0 / (error_sq + c_eps) ** p).detach()
        loss = (weight * (error**2)).mean()

        # For logging purposes
        w = weight.view(B)
        weight_mean = w.mean()
    else:
        loss = F.mse_loss(u_pred, target)
        weight_mean = torch.tensor(0.0, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # 7. Logging Metrics
    # -------------------------------------------------------------------------
    u_pred_rms = (u_pred.to(dtype) ** 2).mean().sqrt()
    target_rms = (target**2).mean().sqrt()
    ratio_r_eq_t = boundary_mask.float().mean()
    ratio_r_neq_t = (~boundary_mask).float().mean()
    ratio_cond_dropped = drop_mask.float().mean()

    logs = {
        "loss": float(loss.item()),
        "raw_mse": float((error**2).mean().item()),
        "u_pred_rms": float(u_pred_rms.item()),
        "target_rms": float(target_rms.item()),
        "weight_mean": float(weight_mean.item()),
        "mean_t": float(t.mean().item()),
        "mean_r": float(r.mean().item()),
        "mean_interval": float((t - r).mean().item()),
        "ratio_r_eq_t": float(ratio_r_eq_t.item()),
        "ratio_r_neq_t": float(ratio_r_neq_t.item()),
        "ratio_cond_dropped": float(ratio_cond_dropped.item()),
        "mean_dt_used": float(dt.mean().item()),
        "cfg_omega": float(omega),
    }

    return loss, logs


@torch.no_grad()
def sample_one_step(
    model: torch.nn.Module,
    edge: torch.Tensor,
    latent_shape: tuple | None = None,
) -> torch.Tensor:
    """Performs one-step generation (1-NFE) using the trained MeanFlow model.

    This function integrates the learned average velocity field from t=1 to t=0
    using a single Euler step, as derived in the MeanFlow framework.

    Args:
        model: Trained MeanFlow model.
        edge: Edge condition [B, 1, H, W].
        latent_shape: Optional tuple for output shape.

    Returns:
        Generated latents [B, 4, H, W].
    """
    device = edge.device
    B = edge.shape[0]

    if latent_shape is None:
        # VAE latent space is 1/8th of the pixel space (256 -> 32)
        H, W = edge.shape[2] // 8, edge.shape[3] // 8
        latent_shape = (B, 4, H, W)

    z1 = torch.randn(latent_shape, device=device, dtype=edge.dtype)
    r = torch.zeros(B, device=device)
    t = torch.ones(B, device=device)

    # Predict average velocity from t=1 to t=0
    u = model(z1, r, t, edge)

    # Euler step: z0 = z1 - (1-0) * u
    # Note: MeanFlow learns the average velocity, so 1 step is theoretically exact
    # if the model has converged perfectly.
    return z1 - u


@torch.no_grad()
def sample_multi_step(
    model: torch.nn.Module,
    edge: torch.Tensor,
    num_steps: int = 2,
    latent_shape: tuple | None = None,
) -> torch.Tensor:
    """Performs multi-step sampling for quality verification.

    While MeanFlow is designed for 1-step generation, multi-step sampling
    can be used to verify the consistency of the learned vector field.

    Args:
        model: Trained MeanFlow model.
        edge: Edge condition [B, 1, H, W].
        num_steps: Number of integration steps.
        latent_shape: Optional tuple for output shape.

    Returns:
        Generated latents [B, 4, H, W].
    """
    device = edge.device
    B = edge.shape[0]

    if latent_shape is None:
        # VAE latent space is 1/8th of the pixel space (256 -> 32)
        H, W = edge.shape[2] // 8, edge.shape[3] // 8
        latent_shape = (B, 4, H, W)

    z = torch.randn(latent_shape, device=device, dtype=edge.dtype)
    time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_curr, t_next = time_steps[i], time_steps[i + 1]

        # Prepare inputs for the current interval
        t = torch.full((B,), float(t_curr), device=device)
        r = torch.full((B,), float(t_next), device=device)

        u = model(z, r, t, edge)

        # Euler update
        z = z - (float(t_curr - t_next)) * u

    return z
