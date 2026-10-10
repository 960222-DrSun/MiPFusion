"""MiPFusion package.

Importing ``models`` registers architectures, datasets, losses and the model into
the official BasicSR registries. ``basicsr`` is used as an external dependency.
"""

from . import arch       # noqa: F401  registers architectures
from . import data       # noqa: F401  registers datasets
from . import losses     # noqa: F401  registers losses
from .mipfusion_model import MiPFusionModel  # noqa: F401  registers the model
from .arch import MiPFusion

__all__ = ['MiPFusion', 'MiPFusionModel']
