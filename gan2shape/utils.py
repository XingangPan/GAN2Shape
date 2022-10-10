import os
import glob
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
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


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def export_to_obj_string(vertices, normal):
    b, h, w, _ = vertices.shape
    vertices[:,:,:,1:2] = -1*vertices[:,:,:,1:2]  # flip y
    vertices[:,:,:,2:3] = 1-vertices[:,:,:,2:3]  # flip and shift z
    vertices *= 100
    vertices_center = nn.functional.avg_pool2d(vertices.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertices = torch.cat([vertices.view(b,h*w,3), vertices_center.view(b,(h-1)*(w-1),3)], 1)

    vertice_textures = get_grid(b, h, w, normalize=True)  # BxHxWx2
    vertice_textures[:,:,:,1:2] = -1*vertice_textures[:,:,:,1:2]  # flip y
    vertice_textures_center = nn.functional.avg_pool2d(vertice_textures.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertice_textures = torch.cat([vertice_textures.view(b,h*w,2), vertice_textures_center.view(b,(h-1)*(w-1),2)], 1) /2+0.5  # Bx(H*W)x2, [0,1]

    vertice_normals = normal.clone()
    vertice_normals[:,:,:,0:1] = -1*vertice_normals[:,:,:,0:1]
    vertice_normals_center = nn.functional.avg_pool2d(vertice_normals.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertice_normals_center = vertice_normals_center / (vertice_normals_center**2).sum(3, keepdim=True)**0.5
    vertice_normals = torch.cat([vertice_normals.view(b,h*w,3), vertice_normals_center.view(b,(h-1)*(w-1),3)], 1)  # Bx(H*W)x2, [0,1]

    idx_map = torch.arange(h*w).reshape(h,w)
    idx_map_center = torch.arange((h-1)*(w-1)).reshape(h-1,w-1)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces2 = torch.stack([idx_map[1:,:w-1], idx_map[1:,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces3 = torch.stack([idx_map[1:,1:], idx_map[:h-1,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces4 = torch.stack([idx_map[:h-1,1:], idx_map[:h-1,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces = torch.cat([faces1, faces2, faces3, faces4], 1)

    objs = []
    mtls = []
    for bi in range(b):
        obj = "# OBJ File:"
        obj += "\n\nmtllib $MTLFILE"
        obj += "\n\n# vertices:"
        for v in vertices[bi]:
            obj += "\nv " + " ".join(["%.4f"%x for x in v])
        obj += "\n\n# vertice textures:"
        for vt in vertice_textures[bi]:
            obj += "\nvt " + " ".join(["%.4f"%x for x in vt])
        obj += "\n\n# vertice normals:"
        for vn in vertice_normals[bi]:
            obj += "\nvn " + " ".join(["%.4f"%x for x in vn])
        obj += "\n\n# faces:"
        obj += "\n\nusemtl tex"
        for f in faces[bi]:
            obj += "\nf " + " ".join(["%d/%d/%d"%(x+1,x+1,x+1) for x in f])
        objs += [obj]

        mtl = "newmtl tex"
        mtl += "\nKa 1.0000 1.0000 1.0000"
        mtl += "\nKd 1.0000 1.0000 1.0000"
        mtl += "\nKs 0.0000 0.0000 0.0000"
        mtl += "\nd 1.0"
        mtl += "\nillum 0"
        mtl += "\nmap_Kd $TXTFILE"
        mtls += [mtl]
    return objs, mtls
