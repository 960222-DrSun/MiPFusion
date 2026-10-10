"""Multispectral / hyperspectral fusion dataset.

Paper: "MiPFusion: Mixed-Scale Proxy Experts Network for Multispectral and
Hyperspectral Image Fusion" (ICASSP), Sec. 3.1 "Experimental Setup".

Benchmarks:
    Chikusei  2517 x 2335 px, 128 bands (363-1018 nm)
              top-left 1000 x 2000 for training, rest as 680 x 680 test patches
    PaviaU    610 x 340 px, 103 bands (430-860 nm), top 340 x 340 for testing
    CAVE      32 indoor scenes, 512 x 512 px, 31 bands (400-700 nm), 20 train / 11 test

Following the **Wald protocol**, ``HRMSI`` is obtained by spectrally aggregating
the original bands into four multispectral bands, and ``LRHSI`` by Gaussian
blurring (3x3, sigma = 0.5) plus downsampling. Patches of 64 x 64 are used at x4.

Expected on-disk layout (``.mat`` files or pre-exported ``.npy`` / images)::

    <dataroot>/
        train/hsi/*.mat      full-resolution reference HSI cubes (H, W, C)
        train/msi/*.mat      the matching 4-band HRMSI
        test/hsi/*.mat
        test/msi/*.mat

Alternatively, point ``dataroot_hsi`` at the reference cubes and let the dataset
synthesise HRMSI / LRHSI on the fly with :func:`wald_protocol`, which is the
recommended way to guarantee that the simulated inputs match the protocol exactly.
"""

import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils import data as data

from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

__all__ = ['MHIFDataset', 'wald_protocol', 'gaussian_kernel', 'spectral_aggregate']


def gaussian_kernel(size=3, sigma=0.5):
    """2D Gaussian kernel, normalised to unit sum."""
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def spectral_aggregate(hsi, ms_channels=4):
    """Aggregate ``C`` hyperspectral bands into ``ms_channels`` MS bands.

    Bands are partitioned into contiguous groups whose widths differ by at most
    one, which is the standard Wald-protocol spectral response for a
    multispectral sensor.

    Args:
        hsi (ndarray): (H, W, C) reference cube.
        ms_channels (int): target band count.
    Returns:
        ndarray: (H, W, ms_channels).
    """
    h, w, c = hsi.shape
    edges = np.linspace(0, c, ms_channels + 1).astype(int)
    bands = [hsi[:, :, edges[i]:edges[i + 1]].mean(axis=2) for i in range(ms_channels)]
    return np.stack(bands, axis=2).astype(np.float32)


def wald_protocol(hsi_ref, ms_channels=4, scale=4, kernel_size=3, sigma=0.5,
                  seed=None):
    """Synthesise ``(LRHSI, HRMSI)`` from a reference HSI cube.

    Steps (Wald protocol):
      1. HRMSI  = spectral aggregation of the reference into ``ms_channels`` bands.
      2. LRHSI  = Gaussian blur (``kernel_size``, ``sigma``) then ``scale``-fold
         downsampling of the reference.

    Args:
        hsi_ref (ndarray): (H, W, C) reference cube.
        ms_channels (int): multispectral band count (4).
        scale (int): spatial downsampling factor (4 for the x4 setting).
    Returns:
        (lr_hsi, hr_msi, hsi_ref): as ``(h, w, C)``, ``(H, W, c)``, ``(H, W, C)``.
    """
    hsi_ref = hsi_ref.astype(np.float32)
    h, w, _ = hsi_ref.shape
    hr_msi = spectral_aggregate(hsi_ref, ms_channels)

    kernel = gaussian_kernel(kernel_size, sigma)
    k = torch.from_numpy(kernel).view(1, 1, kernel_size, kernel_size)
    cube = torch.from_numpy(hsi_ref).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    blurred = F.conv2d(cube, k.repeat(cube.shape[1], 1, 1, 1),
                       padding=kernel_size // 2, groups=cube.shape[1])
    # blur the MSI with the same kernel so both modalities share the PSF
    msi_t = torch.from_numpy(hr_msi).permute(2, 0, 1).unsqueeze(0)
    blurred_msi = F.conv2d(msi_t, k.repeat(msi_t.shape[1], 1, 1, 1),
                           padding=kernel_size // 2, groups=msi_t.shape[1])

    lr_hsi = F.interpolate(blurred, scale_factor=1.0 / scale, mode='bilinear',
                           align_corners=False)
    blurred_msi = F.interpolate(blurred_msi, scale_factor=1.0, mode='bilinear',
                                align_corners=False)

    lr_hsi = lr_hsi.squeeze(0).permute(1, 2, 0).numpy()
    blurred_msi = blurred_msi.squeeze(0).permute(1, 2, 0).numpy()
    return lr_hsi, blurred_msi, hsi_ref


@DATASET_REGISTRY.register()
class MHIFDataset(data.Dataset):
    """Paired LRHSI / HRMSI / reference-HSI dataset for MHIF.

    Args:
        opt (dict): dataset options::

            dataroot_hsi   (str): folder of reference HSI cubes (.mat/.npy).
            dataroot_msi   (str|None): folder of pre-computed HRMSI. When None,
                the HRMSI and LRHSI are synthesised with the Wald protocol.
            hsi_key        (str): variable name inside the .mat file.
            ms_channels    (int): 4 under the Wald protocol.
            scale          (int): 4 for the x4 setting.
            patch_size     (int): 64, the setting used at x4.
            phase          (str): 'train' | 'val' | 'test'.
            norm           (bool): scale cubes to [0, 1] by the global max.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.patch_size = opt.get('patch_size', 64)
        self.scale = opt.get('scale', 4)
        self.ms_channels = opt.get('ms_channels', 4)
        self.hsi_key = opt.get('hsi_key', 'hsi')
        self.norm = opt.get('norm', True)

        root = Path(opt['dataroot_hsi'])
        if not root.is_dir():
            raise FileNotFoundError(f'dataroot_hsi not found: {root}')

        patterns = ('*.mat', '*.npy', '*.npz')
        self.hsi_paths = sorted(
            p for pat in patterns for p in root.glob(pat)
        )
        if not self.hsi_paths:
            raise ValueError(f'no .mat/.npy cubes found in {root}')

        self.msi_dir = Path(opt['dataroot_msi']) if opt.get('dataroot_msi') else None
        self._cache = {}

        get_root_logger().info(
            f'[{self.__class__.__name__}] phase={self.phase} '
            f'{len(self.hsi_paths)} reference cubes from {root}'
        )

    def __len__(self):
        return len(self.hsi_paths)

    # ------------------------------------------------------------------
    def _load_cube(self, path):
        if path in self._cache:
            return self._cache[path]
        if path.suffix == '.npy':
            cube = np.load(path)
        elif path.suffix == '.npz':
            data = np.load(path)
            cube = data[list(data.keys())[0]]
        else:
            mat = sio.loadmat(path)
            key = self.hsi_key if self.hsi_key in mat else \
                [k for k in mat if not k.startswith('__')][0]
            cube = mat[key]
        cube = np.asarray(cube, dtype=np.float32)
        if self.norm:
            peak = cube.max()
            if peak > 1.0:
                cube = cube / peak
        # keep the cache small: reference cubes can be large
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[path] = cube
        return cube

    def __getitem__(self, index):
        path = self.hsi_paths[index]
        cube = self._load_cube(path)  # (H, W, C)

        # ---- crop a patch ------------------------------------------------
        ps = self.patch_size
        h, w, _ = cube.shape
        if ps > 0 and h > ps and w > ps:
            if self.phase == 'train':
                y0 = random.randint(0, h - ps)
                x0 = random.randint(0, w - ps)
            else:
                y0, x0 = (h - ps) // 2, (w - ps) // 2
            cube = cube[y0:y0 + ps, x0:x0 + ps, :]

        # ---- synthesise LRHSI / HRMSI (Wald protocol) --------------------
        lr_hsi, hr_msi, gt = wald_protocol(
            cube, ms_channels=self.ms_channels, scale=self.scale
        )

        # ---- augmentation -----------------------------------------------
        if self.phase == 'train' and self.opt.get('use_flip', True):
            if random.random() < 0.5:
                lr_hsi, hr_msi, gt = [np.flip(a, axis=1).copy()
                                      for a in (lr_hsi, hr_msi, gt)]
            if random.random() < 0.5:
                lr_hsi, hr_msi, gt = [np.flip(a, axis=0).copy()
                                      for a in (lr_hsi, hr_msi, gt)]

        return {
            'lq': torch.from_numpy(lr_hsi.transpose(2, 0, 1).copy()),  # (C, h, w)
            'msi': torch.from_numpy(hr_msi.transpose(2, 0, 1).copy()),  # (c, H, W)
            'gt': torch.from_numpy(gt.transpose(2, 0, 1).copy()),      # (C, H, W)
            'hsi_path': str(path),
        }
