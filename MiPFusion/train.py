"""MiPFusion training entry point.

Registers the project-specific architectures / datasets / losses / model and
delegates the training pipeline to the official ``basicsr.train``.

Usage
-----
    python train.py -opt options/train/MiPFusion_CAVE.yml

    # multi-dataset / multi-band-count override
    python train.py -opt options/train/MiPFusion_CAVE.yml \
        --force_yml datasets.train.dataroot_hsi=/data/PaviaU/train/hsi \
                    datasets.val.dataroot_hsi=/data/PaviaU/test/hsi \
                    network_g.hsi_channels=103

    # 4-GPU distributed
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
        train.py -opt options/train/MiPFusion_CAVE.yml --launcher pytorch
"""

import argparse
import os
import os.path as osp
import sys
from copy import deepcopy

import torch
import yaml

ROOT = osp.dirname(osp.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models  # noqa: E402,F401  registers arch / data / loss / model

from basicsr.train import train_pipeline  # noqa: E402


def _set_nested(opt, keys, value):
    cur = opt
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'],
                        default='none')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--force_yml', nargs='+', default=None,
                        help='override yml keys, e.g. network_g.hsi_channels=103')
    args = parser.parse_args()

    opt_path = args.opt
    if not opt_path.endswith(('.yml', '.yaml')):
        opt_path += '.yml'
    if not osp.isabs(opt_path):
        opt_path = osp.join(root_path, opt_path)
    if not osp.exists(opt_path):
        raise FileNotFoundError(f'option file not found: {opt_path}')

    with open(opt_path, mode='r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    opt['is_train'] = is_train
    opt['name'] = f"{opt['name']}_{opt.get('model_type', 'model')}"

    if args.force_yml is not None:
        for kv in args.force_yml:
            if '=' not in kv:
                raise ValueError(f'--force_yml expects key=value, got {kv!r}')
            key, value = kv.split('=', 1)
            _set_nested(opt, key.split('.'), yaml.safe_load(value))

    if args.debug:
        opt['name'] = 'debug_' + opt['name']
        opt['num_gpu'] = 1
        opt['datasets']['train']['num_worker_per_gpu'] = 0
        opt['datasets']['train']['batch_size_per_gpu'] = 2
        if 'val' in opt['datasets']:
            opt['val']['val_freq'] = 8
        opt['logger']['print_freq'] = 1
        opt['logger']['save_checkpoint_freq'] = 8
        opt['train']['total_epoch'] = 2

    opt['path'] = deepcopy(opt.get('path', {}))
    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        for key in ('models', 'log', 'visualization'):
            os.makedirs(opt['path'][key], exist_ok=True)

        if args.auto_resume and opt['path'].get('resume_state') is None:
            resume_state = osp.join(experiments_root, 'training_states', 'latest.state')
            if osp.exists(resume_state):
                opt['path']['resume_state'] = resume_state
    else:
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
        os.makedirs(opt['path']['visualization'], exist_ok=True)

    opt['dist'] = args.launcher != 'none'
    if opt['dist']:
        opt['rank'] = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        opt['world_size'] = int(os.environ.get('WORLD_SIZE', 1))
    else:
        opt['rank'] = 0
        opt['world_size'] = 1

    return opt


def main():
    opt = parse_options(ROOT, is_train=True)
    train_pipeline(ROOT, opt)


if __name__ == '__main__':
    main()
