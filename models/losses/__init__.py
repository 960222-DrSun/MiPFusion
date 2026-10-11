"""Custom losses for MiPFusion."""

from .composite_loss import (
    SAMLoss, ERGAS, LoadBalancingLoss, PhasedCompositeLoss,
    spectral_angle_mapper, ergas,
)

__all__ = [
    'SAMLoss', 'ERGAS', 'LoadBalancingLoss', 'PhasedCompositeLoss',
    'spectral_angle_mapper', 'ergas',
]
