import os
import glob
import yaml
import random
import numpy as np

import torch
import torch.nn.functional as F


EPS = 1e-7


def setup_runtime(args):
    """Load configs, initialize CuDNN and the random seeds."""

    # Setup CUDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        print(f"Load config from yml file: {args.config}")
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    return cfgs


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b,1,1))**2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b,1,1))**2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    return dist*mask if mask is not None else dist


def smooth_loss(pred_map):
    def gradient(pred):
        if pred.dim() == 4:
            pred = pred.reshape(-1, pred.size(2), pred.size(3))
        D_dy = pred[:, 1:] - pred[:, :-1]
        D_dx = pred[:, :, 1:] - pred[:, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3
    return loss


def photometric_loss(im1, im2, mask=None, conf_sigma=None):
    loss = (im1 - im2).abs()
    if conf_sigma is not None:
        loss = loss *2**0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()
    if mask is not None:
        mask = mask.expand_as(loss)
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss


def resize(image, size):
    dim = image.dim()
    if dim == 3:
        image = image.unsqueeze(1)
    b, _, h, w = image.shape
    if size[0] > h:
        image = F.interpolate(image, size, mode='bilinear')
    elif size[0] < h:
        image = F.interpolate(image, size, mode='area')
    if dim == 3:
        image = image.squeeze(1)
    return image


def crop(tensor, crop_size):
    size = tensor.size(2)   # assume h=w
    margin = (size - crop_size) // 2
    tensor = tensor[:, :, margin:margin+crop_size, margin:margin+crop_size]
    return tensor


def get_mask_range(mask):
    h_range = torch.arange(0, mask.size(0))
    w_range = torch.arange(0, mask.size(1))
    grid = torch.stack(torch.meshgrid([h_range, w_range]), 0).float()
    max_y = torch.max(grid[0, mask])
    min_y = torch.min(grid[0, mask])
    max_x = torch.max(grid[1, mask])
    min_x = torch.min(grid[1, mask])
    return max_y, min_y, max_x, min_x
