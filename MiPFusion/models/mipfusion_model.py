"""MiPFusion model wrapper.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP), Sec. 2.4 and 3.1.

Implements the **phased composite loss** of Eq. (7):

    L = { ||Z_hat - Z||_1 + lambda_sam * L_sam + lambda_bal * L_bal,   e <  10
        { ||Z_hat - Z||_1 + lambda_sam * L_sam,                        e >= 10

with ``lambda_sam = 0.1`` and ``lambda_bal = 0.01``. The load-balancing term is
active only during the first 10 epochs, forcing the routers to distribute traffic
uniformly so that every expert is initialised sufficiently; afterwards it is
removed to avoid interfering with the main-task gradients.

Note that this model consumes **two** inputs (LRHSI and HRMSI), so it does not
use ``SRModel``'s single-input ``feed_data`` pathway unmodified.
"""

import torch
from torch.nn import functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

from models.arch import MiPFusion as MiPFusionArch  # noqa: F401  ensures registration
from models.losses.composite_loss import (
    SAMLoss, LoadBalancingLoss, spectral_angle_mapper, ergas,
)

__all__ = ['MiPFusionModel']


@MODEL_REGISTRY.register()
class MiPFusionModel(SRModel):
    """SRModel specialised for two-input multispectral / hyperspectral fusion.

    ``opt['train']['composite_opt']`` controls the objective::

        train:
          composite_opt:
            l1_weight: 1.0
            lambda_sam: 0.1
            lambda_bal: 0.01
            warmup_epochs: 10
    """

    def init_training_settings(self):
        super().init_training_settings()

        copt = self.opt['train'].get('composite_opt', {})
        self.l1_weight = float(copt.get('l1_weight', 1.0))
        self.sam_loss = SAMLoss(loss_weight=copt.get('lambda_sam', 0.1)).to(self.device)
        self.bal_loss = LoadBalancingLoss(
            loss_weight=copt.get('lambda_bal', 0.01),
            warmup_epochs=copt.get('warmup_epochs', 10),
            phased=True,
        ).to(self.device)

    # ------------------------------------------------------------------
    def feed_data(self, data):
        """Two-input variant: ``lq`` = LRHSI, ``msi`` = HRMSI."""
        self.lq = data['lq'].to(self.device)
        self.msi = data['msi'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def _forward_net(self):
        return self.net_g(self.lq, self.msi)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output = self._forward_net()

        # ---- pixel loss ------------------------------------------------------
        l_pix = self.l1_weight * F.l1_loss(self.output, self.gt)

        # ---- spectral angle mapper (inter-band proportionality) -------------
        l_sam = self.sam_loss(self.output, self.gt)

        # ---- load balancing, active only in the early stage (e < 10) --------
        epoch = int(current_iter // max(self.opt['train'].get('iters_per_epoch', 1), 1))
        gating = []
        if hasattr(self.net_g, 'gating_distributions'):
            gating = self.net_g.gating_distributions()
        l_bal = self.bal_loss(gating, epoch=epoch)

        l_total = l_pix + l_sam
        log = {'l_pix': float(l_pix.detach()), 'l_sam': float(l_sam.detach())}
        if l_bal is not None:
            l_total = l_total + l_bal
            log['l_bal'] = float(l_bal.detach())
        log['l_total'] = float(l_total.detach())

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(log)

    # ------------------------------------------------------------------
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self._forward_net()
        self.net_g.train()

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation reporting PSNR / SSIM / SAM / ERGAS, as in Table 1."""
        metrics = super().nondist_validation(dataloader, current_iter, tb_logger,
                                             save_img)
        if metrics is None:
            metrics = {}

        sam_vals, ergas_vals = [], []
        self.net_g.eval()
        for data in dataloader:
            self.feed_data(data)
            pred = self._forward_net()
            sam_vals.append(float(spectral_angle_mapper(pred, self.gt)))
            ergas_vals.append(float(ergas(pred, self.gt, ratio=self.opt.get('scale', 4))))
        self.net_g.train()

        if sam_vals:
            metrics['sam'] = sum(sam_vals) / len(sam_vals)
        if ergas_vals:
            metrics['ergas'] = sum(ergas_vals) / len(ergas_vals)

        if self.opt.get('rank', 0) == 0:
            message = ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items()
                                if isinstance(v, float))
            if message:
                self.log_dict.update({k: v for k, v in metrics.items()
                                      if isinstance(v, float)})
        return metrics

    @torch.no_grad()
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, **kwargs):
        if self.opt.get('rank', 0) != 0:
            return None
        return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
