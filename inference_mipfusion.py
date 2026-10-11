"""MiPFusion inference: reconstruct an HRHSI from a LRHSI / HRMSI pair.

Usage
-----
    # a single .mat reference cube -> synthesise the Wald-protocol inputs
    python inference_mipfusion.py \
        --checkpoint experiments/MiPFusion_CAVE/models/net_g_latest.pth \
        --hsi  datasets/CAVE/test/hsi/scene01.mat \
        --output results/fusion

    # an explicit LRHSI + HRMSI pair
    python inference_mipfusion.py --checkpoint <ckpt> \
        --hsi .../lr_hsi.mat --msi .../hr_msi.mat --output results/fusion

The script reports PSNR / SSIM / SAM / ERGAS against the reference cube when a
ground truth is available.
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = osp.dirname(osp.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models  # noqa: E402,F401  registers the architectures
from models.arch import MiPFusion  # noqa: E402
from models.data.mhif_dataset import wald_protocol  # noqa: E402
from models.losses.composite_loss import spectral_angle_mapper, ergas  # noqa: E402


def _load_mat(path, key=None):
    if str(path).endswith('.npy'):
        return np.load(path).astype(np.float32)
    mat = sio.loadmat(str(path))
    if key and key in mat:
        return np.asarray(mat[key], dtype=np.float32)
    keys = [k for k in mat if not k.startswith('__')]
    return np.asarray(mat[keys[0]], dtype=np.float32)


def load_network(checkpoint, device, **arch_kwargs):
    net = MiPFusion(**arch_kwargs)
    state = torch.load(checkpoint, map_location='cpu')
    for key in ('params_ema', 'params', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys, first few: {missing[:5]}')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys, first few: {unexpected[:5]}')
    return net.to(device).eval()


def psnr_ssim(pred, gt, data_range=1.0):
    """Band-averaged PSNR and a global SSIM approximation."""
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 10.0 * np.log10(data_range ** 2 / max(mse, 1e-12))
    # simple global SSIM over the flattened cube (for quick sanity checks only;
    # use basicsr.metrics.calculate_ssim for reporting)
    mu_x, mu_y = pred.mean(), gt.mean()
    vx, vy = pred.var(), gt.var()
    cxy = float(np.mean((pred - mu_x) * (gt - mu_y)))
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cxy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2) + 1e-12)
    return psnr, float(ssim)


def main():
    parser = argparse.ArgumentParser(description='MiPFusion inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hsi', type=str, required=True,
                        help='LRHSI cube (when --msi is given), or the '
                             'full-resolution reference cube (when --msi is omitted)')
    parser.add_argument('--msi', type=str, default=None,
                        help='HRMSI; omit to synthesise it with the Wald protocol')
    parser.add_argument('--output', type=str, default='results/fusion')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hsi_channels', type=int, default=31)
    parser.add_argument('--ms_channels', type=int, default=4)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    net = load_network(
        args.checkpoint, device, ms_channels=args.ms_channels,
        hsi_channels=args.hsi_channels, dim=args.dim,
        num_blocks=args.num_blocks, threshold=args.threshold,
        topk=args.topk, scale=args.scale,
    )
    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'MiPFusion loaded: {num_params:.2f} M parameters (paper: 1.39 M)')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = _load_mat(args.hsi)                      # (H, W, C)
    peak = ref.max()
    if peak > 1.0:
        ref = ref / peak

    if args.msi is not None:
        # explicit pair given: --hsi is the LRHSI, --msi the HRMSI
        lr_hsi = _load_mat(args.hsi)
        hr_msi = _load_mat(args.msi)
        if hr_msi.max() > 1.0:
            hr_msi = hr_msi / peak
        gt = None
    else:
        # synthesise the Wald-protocol inputs from the reference cube
        lr_hsi, hr_msi, gt = wald_protocol(
            ref, ms_channels=args.ms_channels, scale=args.scale)

    lq = torch.from_numpy(lr_hsi.transpose(2, 0, 1)[None]).float().to(device)
    msi = torch.from_numpy(hr_msi.transpose(2, 0, 1)[None]).float().to(device)

    with torch.no_grad():
        pred = net(lq, msi)

    # report the sparsity of the sparse cross-scale alignment (Eq. 4)
    sparsity = net.sparsity_report()
    if sparsity['mean_survival_rate'] is not None:
        print(f'sparse routing survival rate: {sparsity["mean_survival_rate"]:.3f} '
              f'over {sparsity["num_levels"]} levels')

    pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_path = out_dir / 'hrhsi_pred.npy'
    np.save(out_path, pred_np)
    print(f'[ok] predicted HRHSI {pred_np.shape} -> {out_path}')

    if gt is not None:
        p, s = psnr_ssim(pred_np, gt)
        p_t = pred
        gt_t = torch.from_numpy(gt.transpose(2, 0, 1)[None]).float().to(device)
        sam = float(spectral_angle_mapper(p_t, gt_t))
        eg = float(ergas(p_t, gt_t, ratio=args.scale))
        print(f'PSNR {p:.4f} dB | SSIM {s:.4f} | SAM {sam:.4f} deg | ERGAS {eg:.4f}')


if __name__ == '__main__':
    main()
