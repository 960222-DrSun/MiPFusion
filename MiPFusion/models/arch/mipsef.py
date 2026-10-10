"""MiPSEF: Mixed Proxy-Scale Expert Fusion.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP), Sec. 2.2 and Fig. 2.

MiPSEF cascades two stages:

**Phase I -- global-local balancing.** The obstacle to multi-scale fusion is the
spatial asynchrony between the local features ``F_c^i`` and the global proxies
``F_c'^i``. Conventional attention produces dense association matrices that bring
noise and overhead. Instead:

1. A **shared encoder** ``E`` maps both inputs into latent codes ``H_l``, ``H_g``,
   decoded by **independent** decoders ``D_l``, ``D_g``. Sharing the encoder
   enforces consistent semantic primitives while the decoders keep detail- and
   prototype-specific projections; matching uses only ``H_l``, ``H_g`` so that it
   is free of decoding bias.
2. Spatial max pooling partitions the maps into ``K`` source and sink blocks, and
   the alignment matrix is ``T = H_{l,p} @ H_{g,p}^T`` (Eq. 3), where ``T_xy``
   measures the matching strength between the y-th global prototype and the x-th
   local block.
3. To enforce sparsity, an element survives only if **both** its row and column
   significance exceed a threshold ``P`` (Eq. 4) -- the local block must have a
   strong reception *and* the global prototype a strong sending tendency.
4. Keeping only the **main diagonal** of ``T'`` imposes a self-alignment
   constraint (``V_k = T'_kk``), assuming index-consistent block pairs. Its
   non-zero entries drive ``G_psi`` (Top-k routing), fused with a learnable
   ``alpha``: ``F_cf = alpha * F_g + (1 - alpha) * F_l`` (Eq. 5).

**Phase II -- spatial-spectral balancing.** Takes ``F_spe^{i-1}`` as a routing
prior for spectral consistency: a ``7x7`` depth-wise convolution with ReLU
captures spatial context, parallel max/average pooling extract multi-granularity
statistics, and a router ``G_Omega`` activates ``k=2`` experts combined into
``F_spe^i``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['MiPSEF', 'SparseCrossScaleRouting', 'TopKRouter']


class TopKRouter(nn.Module):
    """Top-k router over a set of experts, shared by both phases.

    Args:
        dim (int): input channel count.
        num_experts (int): number of experts to route over.
        topk (int): number of activated experts (``k``, default 2 in the paper).
    """

    def __init__(self, dim, num_experts=4, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, num_experts, 1, bias=True),
        )

    def forward(self, x):
        """Returns ``(weights, logits)``; weights is (B, num_experts), rows sum to 1."""
        g = self.avg_pool(x) + self.max_pool(x)
        logits = self.mlp(g).flatten(1)
        topk_val, topk_idx = torch.topk(logits, self.topk, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, topk_idx, F.softmax(topk_val, dim=-1))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights, logits


class SparseCrossScaleRouting(nn.Module):
    """Phase I: saliency-constrained sparse cross-scale routing (Eqs. 3-5).

    Args:
        dim (int): channel count of the local/global features.
        latent_dim (int): width of the shared latent space.
        num_blocks (int): ``K``, number of source/sink blocks.
        threshold (float): significance threshold ``P``.
        topk (int): number of experts selected by ``G_psi``.
    """

    def __init__(self, dim, latent_dim=None, num_blocks=8, threshold=0.5,
                 num_experts=4, topk=2):
        super().__init__()
        latent_dim = latent_dim or dim
        self.K = num_blocks
        self.P = threshold

        # shared encoder E, independent decoders D_l / D_g
        self.encoder = nn.Sequential(
            nn.Conv2d(dim, latent_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1,
                      groups=latent_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder_local = nn.Conv2d(latent_dim, dim, 1, bias=False)
        self.decoder_global = nn.Conv2d(latent_dim, dim, 1, bias=False)

        # learnable fusion weight alpha in Eq. (5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.router = TopKRouter(dim, num_experts, topk)
        self.last_sparsity = None  # diagnostic: fraction of surviving entries

    # ------------------------------------------------------------------
    @staticmethod
    def _block_pool(x, num_blocks):
        """Spatial max pooling into ``num_blocks`` source/sink blocks.

        Returns (B, K, D) by splitting the map into a near-square grid.
        """
        b, d, h, w = x.shape
        # choose a (rows, cols) factorisation closest to the aspect ratio
        best = (1, num_blocks)
        best_err = float('inf')
        for rows in range(1, num_blocks + 1):
            if num_blocks % rows:
                continue
            cols = num_blocks // rows
            err = abs((h / max(rows, 1)) - (w / max(cols, 1)))
            if err < best_err:
                best_err, best = err, (rows, cols)
        rows, cols = best
        # adaptive max pool to the (rows, cols) grid, then flatten to K vectors
        pooled = F.adaptive_max_pool2d(x, (rows, cols))  # (B, D, rows, cols)
        return pooled.flatten(2).transpose(1, 2)         # (B, K, D)

    def forward(self, local, global_proxy):
        """Args:
            local (Tensor): (B, C, H, W) local features ``F_c^i``.
            global_proxy (Tensor): (B, C, H, W) global proxies ``F_c'^i``.

        Returns:
            (Tensor, Tensor, Tensor): ``F_cf``, ``w_g``, ``w_l`` where ``w_g`` /
            ``w_l`` are the scalar weights applied to the global / local branch.
        """
        # shared encoder, independent decoders
        h_l = self.encoder(local)
        h_g = self.encoder(global_proxy)
        f_l = self.decoder_local(h_l)
        f_g = self.decoder_global(h_g)

        # block partitioning of the *latent* codes (matching is decoding-bias free)
        hl_p = self._block_pool(h_l, self.K)   # (B, K, D)
        hg_p = self._block_pool(h_g, self.K)   # (B, K, D)

        # Eq. (3): alignment matrix, T_xy = matching strength
        t = torch.bmm(hl_p, hg_p.transpose(1, 2))  # (B, K, K)

        # Eq. (4): bidirectional significance filtering.
        # The encoder output is post-ReLU, so raw dot products are all positive
        # and scale with the latent width, and -- because both branches share the
        # encoder -- even the normalised cosines sit in a narrow high-similarity
        # cone (~0.97 with sigma ~0.01). An *absolute* threshold P therefore has
        # no discriminating power. Following the paper's intent ("an element
        # survives only if both its row and column significance exceed P"), the
        # significance scores are min-max normalised into [0, 1] per alignment
        # matrix, so that P = 0.5 selects the more significant half of each
        # row / column population. This keeps the survival rate responsive to P.
        hscale = (hl_p.norm(dim=-1, keepdim=True)
                  * hg_p.norm(dim=-1, keepdim=True).transpose(1, 2)).clamp_min(1e-6)
        t_sim = t / hscale                            # (B, K, K) cosine in [-1, 1]

        row_sig = t_sim.mean(dim=-1, keepdim=True)    # reception strength per local block
        col_sig = t_sim.mean(dim=-2, keepdim=True)    # sending tendency per global prototype

        def _normalise(sig):
            lo = sig.amin(dim=(-2, -1), keepdim=True)
            hi = sig.amax(dim=(-2, -1), keepdim=True)
            span = (hi - lo).clamp_min(1e-6)
            return (sig - lo) / span

        row_n, col_n = _normalise(row_sig), _normalise(col_sig)
        mask = (row_n > self.P) & (col_n > self.P)
        t_sparse = t_sim * mask.to(t_sim.dtype)
        self.last_sparsity = float(mask.float().mean().detach())

        # diagonal extraction imposes the self-alignment constraint V_k = T'_kk
        v = torch.diagonal(t_sparse, dim1=-2, dim2=-1)  # (B, K)
        # normalise to a routing prior over the K blocks
        v_vec = F.softmax(v, dim=-1)

        # Eq. (5): Top-k routing over the block prior, then a learnable alpha fusion
        gate_feat = (v_vec.unsqueeze(-1) * hg_p).sum(dim=1)  # (B, D)
        router_in = gate_feat.view(gate_feat.shape[0], -1, 1, 1)
        weights, logits = self.router(router_in)

        alpha = torch.sigmoid(self.alpha)
        fused = alpha * f_g + (1.0 - alpha) * f_l
        return fused, alpha, weights


@ARCH_REGISTRY.register()
class MiPSEF(nn.Module):
    """Mixed Proxy-Scale Expert Fusion module (both phases, Fig. 2).

    Args:
        dim (int): channel count.
        num_blocks (int): ``K`` blocks in the sparse cross-scale routing.
        threshold (float): significance threshold ``P``.
        num_experts (int): expert count in both phases.
        topk (int): ``k=2`` by default.
    """

    def __init__(self, dim, num_blocks=8, threshold=0.5, num_experts=4, topk=2):
        super().__init__()
        self.phase1 = SparseCrossScaleRouting(
            dim, num_blocks=num_blocks, threshold=threshold,
            num_experts=num_experts, topk=topk,
        )

        # ---- Phase II: spatial-spectral balancing ---------------------------
        self.dw7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.router2 = TopKRouter(dim, num_experts, topk)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False),
            ) for _ in range(num_experts)
        ])
        self.norm = nn.GroupNorm(1, dim)
        self.last_gating = None

    def forward(self, local, global_proxy=None, prev_spe=None):
        """Args:
            local (Tensor): (B, C, H, W).
            global_proxy (Tensor | None): (B, C, H, W). Defaults to ``local``.
            prev_spe (Tensor | None): ``F_spe^{i-1}``, the routing prior for
                Phase II. When ``None``, Phase I output is used instead.
        """
        if global_proxy is None:
            global_proxy = local

        # Phase I: global-local balance
        f_cf, alpha, w1 = self.phase1(local, global_proxy)

        # Phase II: spatial-spectral balance, guided by F_spe^{i-1}
        prior = f_cf if prev_spe is None else prev_spe
        # the routing prior comes from the preceding (coarser) scale in the
        # coarse-to-fine loop, so resample it onto this level's grid
        if prior.shape[-2:] != f_cf.shape[-2:]:
            prior = F.interpolate(prior, size=f_cf.shape[-2:],
                                  mode='bilinear', align_corners=False)
        ctx = self.act(self.dw7(f_cf + prior))
        # multi-granularity statistics from parallel max / average pooling
        stats = self.avg_pool(ctx) + self.max_pool(ctx)
        weights, _ = self.router2(stats)
        self.last_gating = weights

        stacked = torch.stack([e(ctx) for e in self.experts], dim=1)
        mixed = (stacked * weights.view(weights.shape[0], -1, 1, 1, 1)).sum(dim=1)

        return self.norm(mixed) + f_cf
