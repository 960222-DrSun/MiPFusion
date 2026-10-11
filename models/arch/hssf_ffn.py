"""HSSF-FFN: Hyperspectral-Spatial Feature Fusion FFN.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP), Sec. 2.4.

HSSF-FFN corrects the residual misalignment between ``F_spa^i`` / ``F_spe^i`` and
the spatially enhanced ``F_se^i``.

* A lightweight branch of ``3x3`` **grouped** convolution and ``7x7``
  **depth-wise separable** convolution enlarges the receptive field at low cost.
* **Channel shuffling** and a **squeeze-and-excitation** block promote
  cross-channel communication for spectral consistency [SENet].
* A **10% spatial random masking** is applied during training so that the network
  learns to infer missing content from its neighbours.
* The refined features are merged **residually** into the spectral basis under
  stable gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['HSSFFFN', 'SEBlock', 'ChannelShuffle']


class ChannelShuffle(nn.Module):
    """Channel shuffle for cross-channel communication."""

    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        if c % g != 0:
            return x
        x = x.view(b, g, c // g, h, w).transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


@ARCH_REGISTRY.register()
class HSSFFFN(nn.Module):
    """Hyperspectral-Spatial Feature Fusion FFN.

    Args:
        dim (int): channel count ``D``.
        group (int): groups for the grouped convolution.
        expansion (float): hidden expansion ratio.
        mask_ratio (float): spatial random masking ratio applied in training
            (the paper uses 10%).
        shuffle_groups (int): groups for channel shuffling.
        use_se (bool): enable the squeeze-and-excitation block.
    """

    def __init__(self, dim, group=4, expansion=2.0, mask_ratio=0.1,
                 shuffle_groups=4, use_se=True):
        super().__init__()
        hidden = int(dim * expansion)
        group = min(group, dim)
        while dim % group:
            group -= 1
        group = max(group, 1)

        # ---- lightweight receptive-field enlargement --------------------------
        self.group_conv = nn.Conv2d(dim, hidden, 3, padding=1, groups=group, bias=False)
        self.dw_sep = nn.Sequential(
            nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
        )
        self.shuffle = ChannelShuffle(shuffle_groups)
        self.act = nn.GELU()
        self.se = SEBlock(hidden) if use_se else nn.Identity()
        self.project = nn.Conv2d(hidden, dim, 1, bias=False)

        self.norm = nn.GroupNorm(1, dim)
        self.mask_ratio = mask_ratio

    def _spatial_mask(self, x):
        """10% spatial random masking during training.

        Whole spatial positions are zeroed across all channels, so the network is
        forced to infer the missing content from neighbouring regions rather than
        from the mask itself.
        """
        if not self.training or self.mask_ratio <= 0:
            return x
        b, _, h, w = x.shape
        # one mask per sample, broadcast over channels
        mask = (torch.rand(b, 1, h, w, device=x.device, dtype=x.dtype)
                > self.mask_ratio).to(x.dtype)
        return x * mask

    def forward(self, x, spectral_basis=None):
        """Args:
            x (Tensor): ``F_se^i``, the spatially enhanced feature to refine.
            spectral_basis (Tensor | None): the spectral basis to merge into
                residually. ``x`` itself is used when not supplied.
        """
        residual = x if spectral_basis is None else spectral_basis

        h = self._spatial_mask(x)
        h = self.act(self.group_conv(h))
        h = self.act(self.dw_sep(h))
        h = self.shuffle(h)
        h = self.se(h)
        h = self.project(h)

        # residual merge into the spectral basis, under stable gradient flow
        return self.norm(h) + residual
