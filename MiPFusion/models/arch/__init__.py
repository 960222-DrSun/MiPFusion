"""MiPFusion architecture registry."""

from .mipsef import MiPSEF, SparseCrossScaleRouting, TopKRouter
from .eg_spamoe import EGSpaMoE, SobelEdge
from .hssf_ffn import HSSFFFN, SEBlock, ChannelShuffle
from .mipfusion import MiPFusion, ProxyScaleGenerator

__all__ = [
    'MiPSEF', 'SparseCrossScaleRouting', 'TopKRouter',
    'EGSpaMoE', 'SobelEdge',
    'HSSFFFN', 'SEBlock', 'ChannelShuffle',
    'MiPFusion', 'ProxyScaleGenerator',
]
