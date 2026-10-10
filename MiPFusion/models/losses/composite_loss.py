"""Losses for MiPFusion.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP), Sec. 2.4, Eq. (7).

The **phased composite loss** is:

    L = { ||Z_hat - Z||_1 + lambda_sam * L_sam + lambda_bal * L_bal,   e <  10
        { ||Z_hat - Z||_1 + lambda_sam * L_sam,                        e >= 10

with ``lambda_sam = 0.1`` and ``lambda_bal = 0.01``.

Design rationale: the load-balancing term is applied **only in the early stage**
(first 10 epochs) to force the routers to distribute traffic uniformly so that
every expert is initialised sufficiently; it is then removed to avoid interfering
with the main-task gradients. The SAM term constrains inter-band proportional
relationships, preserving spectral consistency.

``L_bal`` follows the Switch-Transformer formulation
``L_bal = N * sum_{j=1}^{N} f_j * P_j``, where ``f_j`` is the fraction of tokens
dispatched to expert ``j`` and ``P_j`` the mean router probability for expert
``j``. Its gradient is zero with respect to the dispatch (non-differentiable)
counts while remaining differentiable through the probabilities.

Metrics implemented here (evaluated on the test set, reported in Table 1):
    SAM   -- spectral angle mapper (lower is better)
    ERGAS -- erreur relative globale adimensionnelle de synthese (lower is better)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY

__all__ = [
    'SAMLoss', 'ERGAS', 'LoadBalancingLoss', 'PhasedCompositeLoss',
    'spectral_angle_mapper', 'ergas',
]


def spectral_angle_mapper(pred, target, eps=1e-8):
    """Spectral Angle Mapper, averaged over all pixels.

    Args:
        pred, target (Tensor): (B, C, H, W) or (B, C, N) hyperspectral cubes.
    Returns:
        Tensor: mean angle in **degrees**, in ``[0, 180]``.
    """
    if pred.dim() == 4:
        b, c, h, w = pred.shape
        pred = pred.reshape(b, c, h * w)
        target = target.reshape(b, c, h * w)

    # move the spectral axis last so that the dot product runs over bands
    p = pred.transpose(1, 2)   # (B, N, C)
    t = target.transpose(1, 2)

    # Normalise each spectrum to unit length before the dot product. This makes
    # the cosine exactly 1 for identical (including scaled) spectra, so SAM is 0
    # where it should be. Adding eps *inside* the norm, as a naive
    # ``dot / (|p| |t| + eps)`` does, biases the cosine away from 1 and leaves a
    # spurious residual angle even for a perfect match.
    p_n = F.normalize(p, dim=-1, eps=eps)
    t_n = F.normalize(t, dim=-1, eps=eps)
    cos = (p_n * t_n).sum(dim=-1)

    # The angle is recovered with atan2(sin, cos) instead of acos(cos).
    # acos has an infinite derivative at +-1, so in float32 a perfectly matched
    # spectrum (cos = 1 - 2e-7) still reports ~0.03 deg of spurious angle. The
    # atan2 form is well conditioned over the whole range and gives exactly 0
    # for identical spectra, while remaining scale-invariant.
    #
    # ``sin`` comes from the Gram identity, valid for any band count C:
    #     |p x t|^2 = |p|^2 |t|^2 - (p . t)^2
    # (torch.cross itself only accepts 3-component vectors, which would break
    # for real hyperspectral cubes with tens of bands.)
    dot = (p_n * t_n).sum(dim=-1)

    # In float32 a unit vector only satisfies |p| = 1 to ~1e-7, so an exactly
    # identical pair can yield dot = 1 - 1.8e-7, i.e. ~0.025 deg of pure
    # rounding noise. Anything within that tolerance is physically zero, so it is
    # snapped to an exact match before the angle is taken.
    dot = torch.where(dot.abs() > 1.0 - 1e-6, dot.sign(), dot)
    cross_sq = (1.0 - dot.pow(2)).clamp_min(0.0)
    cross = torch.sqrt(cross_sq)
    angle = torch.atan2(cross, dot)
    return torch.rad2deg(angle).mean()


def ergas(pred, target, ratio=4.0, eps=1e-8):
    """ERGAS (Erreur Relative Globale Adimensionnelle de Synthese).

    Lower is better. Computed over the full cube:

        ERGAS = 100 * (h/l) * sqrt( (1/N) * sum_k (RMSE_k / mu_k)^2 )

    where ``h/l`` is the spatial resolution ratio (``ratio``) and ``mu_k`` the
    mean of band ``k`` of the reference.

    Args:
        pred, target (Tensor): (B, C, H, W) or (B, C, N).
        ratio (float): the spatial ratio ``h/l`` (=4 for the x4 setting).
    """
    if pred.dim() == 4:
        b, c, h, w = pred.shape
        pred = pred.reshape(b, c, h * w)
        target = target.reshape(b, c, h * w)

    diff = pred - target
    rmse_k = diff.pow(2).mean(dim=-1).sqrt()            # (B, C)
    mu_k = target.mean(dim=-1).abs().clamp_min(eps)     # (B, C)
    inner = (rmse_k / mu_k).pow(2).mean(dim=-1)         # (B,)
    return 100.0 * ratio * inner.sqrt().mean()


@LOSS_REGISTRY.register()
class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss (``L_sam``, weight 0.1 in the paper)."""

    def __init__(self, loss_weight=0.1, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * spectral_angle_mapper(pred, target)


@LOSS_REGISTRY.register()
class ERGAS(nn.Module):
    """ERGAS as a differentiable quantity, for validation logging.

    Not part of the training objective in the paper; provided so that the
    reported metric can be tracked during training.
    """

    def __init__(self, loss_weight=1.0, ratio=4.0, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.ratio = ratio

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * ergas(pred, target, ratio=self.ratio)


@LOSS_REGISTRY.register()
class LoadBalancingLoss(nn.Module):
    """Switch-Transformer load balancing, ``L_bal = N * sum_j f_j * P_j``.

    Args:
        loss_weight (float): ``lambda_bal``, 0.01 in the paper.
        num_experts (int or None): ``N``. Inferred from the gating tensors when
            None, which requires them to be the full ``(B, n)`` probabilities.
        phased (bool): when True, the term is only active for the first
            ``warmup_epochs`` epochs; afterwards it returns 0. This mirrors
            "it is removed later to avoid interfering with the main-task
            gradients".
        warmup_epochs (int): the paper's early stage length, 10 epochs.
    """

    def __init__(self, loss_weight=0.01, num_experts=None, phased=True,
                 warmup_epochs=10, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.phased = phased
        self.warmup_epochs = warmup_epochs

    def forward(self, gating_list, epoch=None, weight=None, **kwargs):
        """Args:
            gating_list: list of ``(B, n)`` router probability tensors.
            epoch (int | None): current epoch, used by the phased schedule.
        """
        if self.phased and epoch is not None and epoch >= self.warmup_epochs:
            return None  # term disabled in the later stage (Eq. 7, e >= 10)

        if not gating_list:
            return None

        terms = []
        for g in gating_list:
            if g is None or g.dim() != 2:
                continue
            n = self.num_experts or g.shape[1]
            probs = g.clamp_min(1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # P_j: mean router probability for expert j
            p_j = probs.mean(dim=0)
            # f_j: fraction dispatched to expert j (non-differentiable counts)
            with torch.no_grad():
                top1 = probs.argmax(dim=-1)
                counts = torch.bincount(top1, minlength=n).to(probs.dtype)
                f_j = counts / max(top1.numel(), 1)
            terms.append(n * (f_j * p_j).sum())

        if not terms:
            return None
        return self.loss_weight * torch.stack(terms).mean()


@LOSS_REGISTRY.register()
class PhasedCompositeLoss(nn.Module):
    """The full phased composite objective of Eq. (7).

        train:
          composite_opt:
            type: PhasedCompositeLoss
            l1_weight: 1.0
            lambda_sam: 0.1
            lambda_bal: 0.01
            warmup_epochs: 10
    """

    def __init__(self, l1_weight=1.0, lambda_sam=0.1, lambda_bal=0.01,
                 warmup_epochs=10, **kwargs):
        super().__init__()
        self.l1_weight = l1_weight
        self.sam = SAMLoss(loss_weight=lambda_sam)
        self.bal = LoadBalancingLoss(loss_weight=lambda_bal,
                                     warmup_epochs=warmup_epochs)
        self.loss_dict = {}

    def forward(self, pred, target, gating_list=None, epoch=None, **kwargs):
        l_pix = self.l1_weight * F.l1_loss(pred, target)
        l_sam = self.sam(pred, target)
        l_bal = self.bal(gating_list or [], epoch=epoch)

        total = l_pix + l_sam
        bal_val = 0.0
        if l_bal is not None:
            total = total + l_bal
            bal_val = float(l_bal.detach())

        self.loss_dict = {
            'l_pix': float(l_pix.detach()),
            'l_sam': float(l_sam.detach()),
            'l_bal': bal_val,
            'balanced_active': l_bal is not None,
        }
        return total
