"""MiPFusion: Mixed-Scale Proxy Experts Network for MHIF.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP). Code: https://github.com/960222-DrSun/MiPFusion

Given ``LRHSI X in R^{h x w x C}`` and ``HRMSI Y in R^{H x W x c}``, learn a
mapping ``M(.)`` reconstructing ``Z_hat in R^{H x W x C}``, i.e. a
high-resolution hyperspectral image with both high spatial resolution and high
spectral fidelity (Eq. 1).

Pipeline (Fig. 1):
    LRHSI, HRMSI --Head--> (a) Proxy Scale Generator (HRNet-style, 3 levels)
                            (b) MiPSEF -> EG-SpaMoE -> HSSF-FFN, per level
                            --Tail--> HRHSI

Reporting targets (paper, Table 1, 1.39 M params, 5.12 ms latency at x4):
    Chikusei  PSNR / SSIM / SAM / ERGAS   best on every metric
    CAVE      +0.30 dB over the second-best method
    PaviaU    +0.62 dB over the second-best method

Training recipe: ``D=64``, AdamW lr ``4e-4``, wd ``1e-2``, batch 8, 300 epochs,
``lambda_sam=0.1``, ``lambda_bal=0.01`` with the load-balancing term active only
for the first 10 epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .mipsef import MiPSEF
from .eg_spamoe import EGSpaMoE
from .hssf_ffn import HSSFFFN

__all__ = ['MiPFusion', 'ProxyScaleGenerator']


class ConvBlock(nn.Module):
    """Standard conv-GELU-conv residual block used inside the generator."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.body(x)


class ProxyScaleGenerator(nn.Module):
    """(a) Proxy Scale Generator -- HRNet-style, high resolution throughout.

    Following HRNet, two streams are kept over three levels ``i = 1, 2, 3``
    (Eq. 2):

        F_c^i  = E_c(down_2(F_c^{i-1}))      i = 2, 3    local features
        F_c'^i = E_c'(up_2(F_c'^{i+1}))      i = 1, 2    global proxies

    with ``F_c^1`` from the input HRMSI and ``F_c'^3`` from ``F_c^3``. Keeping
    high resolution throughout avoids the information loss of bottlenecks and
    decouples the two priors before re-alignment.
    """

    def __init__(self, dim=64, ms_channels=4, hsi_channels=31, num_levels=3):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # separate stems: MSI carries the spatial detail, HSI the spectrum
        self.head_msi = nn.Conv2d(ms_channels, dim, 3, padding=1, bias=False)
        self.head_hsi = nn.Conv2d(hsi_channels, dim, 3, padding=1, bias=False)

        self.local_blocks = nn.ModuleList([ConvBlock(dim) for _ in range(num_levels)])
        self.global_blocks = nn.ModuleList([ConvBlock(dim) for _ in range(num_levels)])

        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False), nn.GELU()
            ) for _ in range(num_levels - 1)
        ])
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
            ) for _ in range(num_levels - 1)
        ])

    def forward(self, x_hsi, y_msi):
        """Args:
            x_hsi (Tensor): LRHSI, (B, C, h, w).
            y_msi (Tensor): HRMSI, (B, c, H, W).

        Returns:
            (list, list): ``locals_`` = [F_c^1, F_c^2, F_c^3] (fine -> coarse),
                          ``globals_`` = [F_c'^1, F_c'^2, F_c'^3] (fine -> coarse).
        """
        # local stream: progressive downsampling, starting from the HRMSI
        local = self.head_msi(y_msi)
        locals_ = []
        for i in range(self.num_levels):
            if i > 0:
                local = self.down[i - 1](local)
            local = self.local_blocks[i](local)
            locals_.append(local)

        # the LRHSI is upsampled to the HR grid to seed the global proxy branch
        hsi_up = F.interpolate(x_hsi, size=y_msi.shape[-2:], mode='bilinear',
                               align_corners=False)
        global_feat = self.head_hsi(hsi_up) + locals_[0]
        global_feat = self.global_blocks[-1](global_feat)

        # global stream: progressive upsampling, seeded from the coarsest local
        globals_ = [None] * self.num_levels
        globals_[-1] = global_feat
        for i in range(self.num_levels - 2, -1, -1):
            global_feat = self.up[i](global_feat)
            global_feat = self.global_blocks[i](global_feat)
            globals_[i] = global_feat

        return locals_, globals_


@ARCH_REGISTRY.register()
class MiPFusion(nn.Module):
    """Mixed-Scale Proxy Experts Network for multispectral / hyperspectral fusion.

    Args:
        ms_channels (int): HRMSI band count ``c`` (4 under the Wald protocol).
        hsi_channels (int): LRHSI band count ``C`` (e.g. 31 for CAVE, 103 for
            PaviaU, 128 for Chikusei).
        dim (int): feature width ``D``. The paper uses ``D = 64``.
        num_levels (int): ``L``, generator levels. Default: 3.
        num_blocks (int): ``K`` blocks in the sparse cross-scale routing.
        threshold (float): significance threshold ``P`` in Eq. (4).
        num_experts (int): expert pool size in MiPSEF / EG-SpaMoE.
        topk (int): ``k=2`` activated experts.
        scale (int): spatial scaling factor ``s = H/h``. Default: 4 (x4).
        use_mipsef / use_eg_spamoe / use_hssf (bool): component switches, used to
            reproduce the component ablation.
    """

    def __init__(self, ms_channels=4, hsi_channels=31, dim=64, num_levels=3,
                 num_blocks=8, threshold=0.5, num_experts=4, topk=2, scale=4,
                 use_mipsef=True, use_eg_spamoe=True, use_hssf=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.hsi_channels = hsi_channels
        self.scale = scale
        self.use_mipsef = use_mipsef
        self.use_eg_spamoe = use_eg_spamoe
        self.use_hssf = use_hssf

        self.generator = ProxyScaleGenerator(
            dim=dim, ms_channels=ms_channels, hsi_channels=hsi_channels,
            num_levels=num_levels,
        )

        # ---- MiPSEF: cross-scale alignment, one per level --------------------
        if use_mipsef:
            self.mipsef = nn.ModuleList([
                MiPSEF(dim, num_blocks=num_blocks, threshold=threshold,
                       num_experts=num_experts, topk=topk)
                for _ in range(num_levels)
            ])
        else:
            self.mipsef = None

        # ---- EG-SpaMoE: edge-guided spatial detail recovery ------------------
        if use_eg_spamoe:
            self.eg_spamoe = nn.ModuleList([
                EGSpaMoE(dim, num_experts=num_experts, topk=topk,
                         level=i + 1, num_levels=num_levels)
                for i in range(num_levels)
            ])
        else:
            self.eg_spamoe = None

        # ---- HSSF-FFN: spectral consistency ---------------------------------
        if use_hssf:
            self.hssf = nn.ModuleList([HSSFFFN(dim) for _ in range(num_levels)])
        else:
            self.hssf = None

        # ---- tail: map the fused feature to the HRHSI band count -------------
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, hsi_channels, 3, padding=1, bias=False),
        )

        # the LRHSI is upsampled and added residually, so the network predicts
        # only the high-frequency residual it needs to recover
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    def forward(self, x_hsi, y_msi):
        """Args:
            x_hsi (Tensor): LRHSI, (B, C, h, w).
            y_msi (Tensor): HRMSI, (B, c, H, W).

        Returns:
            Tensor: reconstructed HRHSI, (B, C, H, W).
        """
        target_size = y_msi.shape[-2:]
        hsi_up = F.interpolate(x_hsi, size=target_size, mode='bilinear',
                               align_corners=False)

        locals_, globals_ = self.generator(x_hsi, y_msi)

        # process coarse -> fine so that F_spe^{i-1} is available as the routing
        # prior of level i (Phase II of MiPSEF), and the cross-stage partner
        # F_spe^{L-i+1} is available to EG-SpaMoE.
        F_spe = [None] * self.num_levels
        for i in range(self.num_levels - 1, -1, -1):
            local_i = locals_[i]
            global_i = globals_[i]

            # local and global streams may differ in resolution at coarse levels;
            # align the proxy to the local grid before fusion
            if global_i.shape[-2:] != local_i.shape[-2:]:
                global_i = F.interpolate(global_i, size=local_i.shape[-2:],
                                         mode='bilinear', align_corners=False)

            # ---- MiPSEF -----------------------------------------------------
            if self.mipsef is not None:
                # Phase II takes F_spe^{i-1} as the routing prior. The loop runs
                # coarse -> fine, so ``F_spe[i+1]`` is the previously computed
                # (coarser) level and must be resampled onto this level's grid.
                prev = F_spe[i + 1] if i + 1 < self.num_levels else None
                if prev is not None and prev.shape[-2:] != local_i.shape[-2:]:
                    prev = F.interpolate(prev, size=local_i.shape[-2:],
                                         mode='bilinear', align_corners=False)
                fe = self.mipsef[i](local_i, global_i, prev_spe=prev)
            else:
                fe = local_i + global_i

            # ---- EG-SpaMoE: partner is level (L - i + 1), 1-based -----------
            if self.eg_spamoe is not None:
                partner_idx = self.num_levels - (i + 1)  # 0-based partner
                partner = F_spe[partner_idx] if partner_idx != i else None
                if partner is not None and partner.shape[-2:] != fe.shape[-2:]:
                    partner = F.interpolate(partner, size=fe.shape[-2:],
                                            mode='bilinear', align_corners=False)
                f_spa = fe + self.eg_spamoe[i](fe, partner)
            else:
                f_spa = fe

            # ---- HSSF-FFN ---------------------------------------------------
            if self.hssf is not None:
                F_spe[i] = self.hssf[i](f_spa, spectral_basis=local_i)
            else:
                F_spe[i] = f_spa

        # the finest level carries the full-resolution fused representation
        fused = F_spe[0]
        if fused.shape[-2:] != target_size:
            fused = F.interpolate(fused, size=target_size, mode='bilinear',
                                  align_corners=False)

        # residual merge onto the upsampled LRHSI, weighted by a learnable scalar
        out = hsi_up + self.residual_weight * self.tail(fused)
        return out

    # ------------------------------------------------------------------
    def gating_distributions(self):
        """Collect per-layer router weights for the load-balancing term.

        Returns a list of ``(B, n)`` tensors, one per MiPSEF / EG-SpaMoE router.
        The paper's ``L_bal = N * sum_j f_j P_j`` forces the routers to distribute
        traffic uniformly so that every expert is initialised sufficiently.
        """
        gating = []
        for module_list in (self.mipsef, self.eg_spamoe):
            if module_list is None:
                continue
            for layer in module_list:
                if getattr(layer, 'last_gating', None) is not None:
                    gating.append(layer.last_gating)
        return gating

    def sparsity_report(self):
        """Diagnostic: mean survival rate of the sparse alignment matrix.

        ``MiPSEF.last_sparsity`` records the fraction of ``T'`` entries that
        passed the bidirectional significance filter of Eq. (4). A healthy value
        is clearly below 1.0, confirming that the routing is genuinely sparse.
        """
        rates = []
        if self.mipsef is not None:
            for layer in self.mipsef:
                rate = getattr(layer.phase1, 'last_sparsity', None)
                if rate is not None:
                    rates.append(rate)
        return {
            'mean_survival_rate': float(sum(rates) / len(rates)) if rates else None,
            'num_levels': len(rates),
        }
