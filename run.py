import argparse

import torch
import torch.distributed as dist
from mmcv.runner import init_dist

from gan2shape import setup_runtime, Trainer, GAN2Shape


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--launcher', default=None, type=str, help='Launcher')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)

if args.launcher is None or args.launcher == 'none':
    cfgs['distributed'] = False
else:
    cfgs['distributed'] = True
    init_dist(args.launcher, backend='nccl')
    # important: use different random seed for different process
    torch.manual_seed(args.seed + dist.get_rank())

print(cfgs)
trainer = Trainer(cfgs, GAN2Shape)

## run
trainer.train()