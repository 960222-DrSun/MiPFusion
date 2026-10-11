"""EG-SpaMoE: Edge-Guided Spatial Mixture of Experts.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP), Sec. 2.3.

EG-SpaMoE recovers spatial structure using **Sobel edge priors**. An edge map
``F_edg`` modulates the spatial feature ``F_spa^i`` and the complementary
cross-stage feature ``F_spe^{L-i+1}`` into two edge-enhanced streams (Eq. 6):

    F_up^i  = F_edg * DWConv7x7(F_spa^i)
    F_low^i = F_edg * DWConv7x7(F_spe^{L-i+1})

* The **upper** branch extracts long-range dependencies with ``7x7`` depth-wise
  plus ``3x3`` standard convolutions.
* The **lower** branch uses ``3x3`` depth-wise plus ``3x3`` standard convolutions
  for local structures.

A router ``G_omega`` selects the ``k=2`` most relevant spatial / spatial-spectral
experts from the concatenated features; their weighted outputs are refined by a
channel calibration module (``1x1`` conv, LayerNorm, ReLU) and added residually
into ``F_spa^{i+1}``. Edge-aware modulation plus sparse selection therefore
preserves boundaries that plain convolutions smooth away.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['EGSpaMoE', 'SobelEdge']


class SobelEdge(nn.Module):
    """Fixed Sobel operator producing a single-channel normalised edge map.

    The kernels are registered as buffers so they move with the module but are
    never updated by the optimiser.

    Normalisation note: a per-image min-max stretch would map a *flat* region to
    an all-ones map (min == max -> the denominator is clamped and the numerator
    collapses to 0/eps), destroying the "no edge" signal. The magnitude is
    therefore scaled by a fixed, data-independent constant and clamped to
    ``[0, 1]``, which keeps flat regions at 0 and only saturates genuine edges.
    """

    def __init__(self, in_channels=1, normalize=True):
        super().__init__()
        self.normalize = normalize
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('kx', kx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer('ky', ky.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        """x: (B, C, H, W) -> (B, 1, H, W) in [0, 1]."""
        gx = F.conv2d(x, self.kx, padding=1, groups=x.shape[1])
        gy = F.conv2d(x, self.ky, padding=1, groups=x.shape[1])
        # mean over channels keeps the map single-channel and channel-agnostic
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6).mean(dim=1, keepdim=True)
        if self.normalize:
            # Sobel magnitude of a unit step edge is 4 * 2 = 8 per channel; scale
            # by a fixed constant rather than by the per-image range.
            mag = (mag / 4.0).clamp(0.0, 1.0)
        return mag


@ARCH_REGISTRY.register()
class EGSpaMoE(nn.Module):
    """Edge-Guided Spatial Mixture of Experts.

    Args:
        dim (int): channel count ``D``.
        num_experts (int): expert pool size.
        topk (int): ``k=2`` experts activated.
        level (int): 1-based scale index ``i``. With ``L`` levels, the branch
            partner at level ``i`` is ``L - i + 1``.
        num_levels (int): total number of levels ``L``. Default: 3.
    """

    def __init__(self, dim, num_experts=4, topk=2, level=1, num_levels=3):
        super().__init__()
        self.dim = dim
        self.level = level
        self.num_levels = num_levels
        self.topk = min(topk, num_experts)

        self.edge = SobelEdge(in_channels=dim)

        # ---- two edge-enhanced streams (Eq. 6) -------------------------------
        self.dw7_spa = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.dw7_spe = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)

        # upper branch: long-range (7x7 DWConv + 3x3 standard conv)
        self.up_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
        )
        # lower branch: local (3x3 DWConv + 3x3 standard conv)
        self.low_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
        )

        # ---- router G_omega over spatial / spatial-spectral experts -----------
        self.router = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_experts, 1, bias=True),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 3, padding=1, groups=2, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False),
            ) for _ in range(num_experts)
        ])

        # ---- channel calibration module --------------------------------------
        self.calibrate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.ReLU(inplace=True),
        )
        self.last_gating = None

    def forward(self, f_spa, f_spe_stage=None):
        """Args:
            f_spa (Tensor): ``F_spa^i``, the spatial feature at this level.
            f_spe_stage (Tensor | None): the complementary cross-stage feature
                ``F_spe^{L-i+1}``. When ``None`` (e.g. ``L=1``), ``f_spa`` is
                reused so that the two-stream modulation degenerates gracefully.
        """
        if f_spe_stage is None:
            f_spe_stage = f_spa

        # actual partner index, clamped to the valid level range
        partner = self.num_levels - self.level + 1

        # Eq. (6): edge-modulated two-stream features
        edge = self.edge(f_spa)                     # (B, 1, H, W)
        f_up = edge * self.dw7_spa(f_spa)
        f_low = edge * self.dw7_spe(f_spe_stage)

        upper = self.up_branch(f_up)
        lower = self.low_branch(f_low)
        concat = torch.cat([upper, lower], dim=1)   # (B, 2D, H, W)

        # sparse expert selection
        pooled = F.adaptive_avg_pool2d(concat, 1) + F.adaptive_max_pool2d(concat, 1)
        logits = self.router(pooled).flatten(1)
        topk_val, topk_idx = torch.topk(logits, self.topk, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, topk_idx, F.softmax(topk_val, dim=-1))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.last_gating = weights

        stacked = torch.stack([e(concat) for e in self.experts], dim=1)
        mixed = (stacked * weights.view(weights.shape[0], -1, 1, 1, 1)).sum(dim=1)

        return self.calibrate(mixed)
