import os
import math
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision
from torchvision import transforms

from . import networks
from . import utils
from .stylegan2 import Generator, Discriminator, PerceptualLoss
from .renderer import Renderer
from .losses import DiscriminatorLoss


def map_func(storage, location):
    return storage.cpu()


class GAN2Shape():
    def __init__(self, cfgs):
        # basic parameters
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.distributed = cfgs.get('distributed')
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.mode = 'step1'
        self.category = cfgs.get('category', 'face')
        self.cfgs = cfgs
        # functional parameters
        self.joint_train = cfgs.get('joint_train', False)
        self.independent = cfgs.get('independent', True)
        self.share_weight = cfgs.get('share_weight', True)
        self.relative_enc = cfgs.get('relative_enc', False)
        self.use_mask = cfgs.get('use_mask', True)
        self.add_mean_L = cfgs.get('add_mean_L', False)
        self.add_mean_V = cfgs.get('add_mean_V', False)
        self.flip1 = cfgs.get('flip1_cfg', [False])[0]
        self.flip3 = cfgs.get('flip3_cfg', [False])[0]
        self.reset_weight = cfgs.get('reset_weight', False)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        # detailed parameters
        self.image_size = cfgs.get('image_size', 128)
        self.crop = cfgs.get('crop', None)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.view_scale = cfgs.get('view_scale', 1.0)
        self.collect_iters = cfgs.get('collect_iters', 100)
        self.rand_light = cfgs.get('rand_light', [-1,1,-0.2,0.8,-0.1,0.6,-0.6])
        # optimization parameters
        self.batchsize = cfgs.get('batchsize', 8)
        self.lr = cfgs.get('lr', 1e-4)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_smooth = cfgs.get('lam_smooth', 0.01)
        self.lam_regular = cfgs.get('lam_regular', 0.01)
        # StyleGAN parameters
        self.channel_multiplier = cfgs.get('channel_multiplier', 2)
        self.gan_size = cfgs.get('gan_size', 256)
        self.z_dim = cfgs.get('z_dim', 512)
        self.truncation = cfgs.get('truncation', 1)
        self.F1_d = cfgs.get('F1_d', 2)
        # networks and optimizers
        self.generator = Generator(self.gan_size, self.z_dim, 8, channel_multiplier=self.channel_multiplier)
        self.discriminator = Discriminator(self.gan_size, channel_multiplier=self.channel_multiplier)

        gn_base = 8 if self.image_size >= 128 else 16
        nf = max(4096 // self.image_size, 16)
        self.netD = networks.EDDeconv(cin=3, cout=1, size=self.image_size, nf=nf, gn_base=gn_base, zdim=256, activation=None)
        self.netA = networks.EDDeconv(cin=3, cout=3, size=self.image_size, nf=nf, gn_base=gn_base, zdim=256)
        self.netV = networks.Encoder(cin=3, cout=6, size=self.image_size, nf=nf)
        self.netL = networks.Encoder(cin=3, cout=4, size=self.image_size, nf=nf)
        self.netEnc = networks.ResEncoder(3, 512, size=self.image_size, nf=32, activation=None)
        self.network_names = [k for k in vars(self) if 'net' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        self.PerceptualLoss = PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=True, gpu_ids=[torch.device(self.rank)]
        )
        self.d_loss = DiscriminatorLoss(ftr_num=4)
        self.renderer = Renderer(cfgs, self.image_size)

        # depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d: (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.depth_inv_rescaler = lambda d: (d-self.min_depth) / (self.max_depth-self.min_depth)  # (min_depth,max_depth) => (0,1)

        # light and viewpoint sampler
        self.init_VL_sampler()

        # load pre-trained weights
        ckpt = cfgs.get('pretrain', None)
        if ckpt is not None:
            self.ckpt = torch.load(ckpt, map_location=map_func)
            self.load_model_state(self.ckpt)
        gan_ckpt = torch.load(cfgs.get('gan_ckpt'), map_location=map_func)
        self.generator.load_state_dict(gan_ckpt['g_ema'], strict=False)
        self.generator = self.generator.cuda()
        self.generator.eval()
        self.discriminator.load_state_dict(gan_ckpt['d'], strict=False)
        self.discriminator = self.discriminator.cuda()
        self.discriminator.eval()
        if self.truncation < 1:
            with torch.no_grad():
                self.mean_latent = self.generator.mean_latent(4096)
        else:
            self.mean_latent = None

        for k in self.network_names:
            network = getattr(self, k)
            network = network.cuda()
        # distributed
        if self.distributed and self.share_weight:
            for net_name in self.network_names:
                setattr(self, net_name, DDP(getattr(self, net_name),
                                            device_ids=[torch.cuda.current_device()],
                                            find_unused_parameters=True))

        self.need_ellipsoid_init = False
        if ckpt is None or 'netD' not in self.ckpt.keys():
            self.need_ellipsoid_init = True

        if not hasattr(self, 'ckpt'):
            # copy model weights, used for reseting weights
            self.ckpt = deepcopy(self.get_model_state())

        self.init_parsing_model()  # initialize image parsing models
        self.proj_idx = 0  # index used for loading projected samples
        self.canon_mask = None

    def reset_model_weight(self):
        self.load_model_state(self.ckpt)

    def setup_target(self, image_path, gt_depth_path, latent_path):
        self.image_path = image_path
        self.gt_depth_path = gt_depth_path
        self.w_path = latent_path
        self.load_data()
        self.load_latent()
        # prepare object mask, used for shape initialization and optionally remove the background
        self.prepare_mask()
        if self.need_ellipsoid_init:
            self.init_netD_ellipsoid()
            if not self.reset_weight:
                self.need_ellipsoid_init = False

    def load_data(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        def load_depth(depth_path):
            depth_gt = Image.open(depth_path)
            if self.crop is not None:
                depth_gt = transforms.CenterCrop(self.crop)(depth_gt)
            depth_gt = transform(depth_gt).cuda()
            depth_gt = (1 - depth_gt) * 2 - 1
            depth_gt = self.depth_rescaler(depth_gt)
            return depth_gt

        def load_image(image_path):
            image = Image.open(image_path)
            self.origin_size = image.size[0]  # we assume h=w
            if self.crop is not None:
                image = transforms.CenterCrop(self.crop)(image)
            image = transform(image).unsqueeze(0).cuda()
            image = image * 2 - 1
            return image

        if self.joint_train:
            assert type(self.image_path) is list
            self.input_im_all = []
            if self.load_gt_depth:
                self.depth_gt_all = []
            self.img_num = len(self.image_path)
            assert self.collect_iters >= self.img_num
            print("Loading images...")
            for i in range(self.img_num):
                image_path = self.image_path[i]
                input_im = load_image(image_path)
                self.input_im_all.append(input_im.cpu())
                if self.load_gt_depth:
                    depth_path = self.gt_depth_path[i]
                    depth_gt = load_depth(depth_path)
                    self.depth_gt_all.append(depth_gt.cpu())
            self.input_im = self.input_im_all[0].cuda()
            if self.load_gt_depth:
                self.depth_gt = self.depth_gt_all[0].cuda()
            # img_idx is used to track the index of current image
            self.img_idx = 0
            self.idx_perm = torch.LongTensor(list(range(self.img_num)))
        else:
            if type(self.image_path) is list:
                assert len(self.image_path) == self.world_size
                self.image_path = self.image_path[self.rank]
                if self.load_gt_depth:
                    self.gt_depth_path = self.gt_depth_path[self.rank]
            print("Loading images...")
            self.input_im = load_image(self.image_path)
            if self.load_gt_depth:
                self.depth_gt = load_depth(self.gt_depth_path)

    def load_latent(self):
        with torch.no_grad():
            def get_w_img(w_path):
                latent_w = torch.load(w_path, map_location='cpu')
                if type(latent_w) is dict:
                    latent_w = latent_w['latent']
                if latent_w.dim() == 1:
                    latent_w = latent_w.unsqueeze(0)
                latent_w = latent_w.cuda()

                gan_im, _ = self.generator([latent_w], input_is_w=True, truncation_latent=self.mean_latent,
                                           truncation=self.truncation, randomize_noise=False)
                gan_im = gan_im.clamp(min=-1, max=1)
                if self.crop is not None:
                    gan_im = utils.resize(gan_im, [self.origin_size, self.origin_size])
                    gan_im = utils.crop(gan_im, self.crop)
                gan_im = utils.resize(gan_im, [self.image_size, self.image_size])
                return latent_w, gan_im

            if self.joint_train:
                assert type(self.w_path) is list
                self.latent_w_all, self.gan_im_all = [], []
                for w_path in self.w_path:
                    latent_w, gan_im = get_w_img(w_path)
                    self.latent_w_all.append(latent_w.cpu())
                    self.gan_im_all.append(gan_im.cpu())
                self.latent_w = self.latent_w_all[0].cuda()
                self.gan_im = self.gan_im_all[0].cuda()
            else:
                if type(self.w_path) is list:
                    assert len(self.w_path) == self.world_size
                    self.w_path = self.w_path[self.rank]
                self.latent_w, self.gan_im = get_w_img(self.w_path)

            self.center_w = self.generator.style_forward(torch.zeros(1, self.z_dim).cuda())
            self.center_h = self.generator.style_forward(torch.zeros(1, self.z_dim).cuda(), depth=8-self.F1_d)

    def prepare_mask(self):
        with torch.no_grad():
            if self.joint_train:
                self.input_mask_all = []
                for i in range(self.img_num):
                    input_im = self.input_im_all[i].cuda()
                    self.input_mask_all.append(self.parse_mask(input_im).cpu())
                self.input_mask = self.input_mask_all[0].cuda()
            else:
                self.input_mask = self.parse_mask(self.input_im)

    def next_image(self):
        # Used in joint training mode
        self.img_idx += 1
        if self.img_idx >= self.img_num:
            self.img_idx = 0
            rand_idx = torch.randperm(self.img_num)  # shuffle
            self.idx_perm = self.idx_perm[rand_idx]
        idx = self.idx_perm[self.img_idx].item()
        self.input_im = self.input_im_all[idx].cuda()
        self.input_mask = self.input_mask_all[idx].cuda()
        self.latent_w = self.latent_w_all[idx].cuda()
        self.gan_im = self.gan_im_all[idx].cuda()
        if self.load_gt_depth:
            self.depth_gt = self.depth_gt_all[idx].cuda()
        if self.mode == 'step2':
            self.depth = self.depth_all[idx].cuda()
            self.albedo = self.albedo_all[idx].cuda()
            self.light = self.light_all[idx].cuda()
            self.normal = self.normal_all[idx].cuda()
            if self.use_mask:
                self.canon_mask = self.canon_mask_all[idx].cuda()

    def init_netD_ellipsoid(self):
        ellipsoid = self.init_ellipsoid()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.netD.parameters()),
            lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)

        print("Initializing the depth net to output ellipsoid ...")
        for i in range(1000):
            depth_raw = self.netD(self.input_im)
            depth = depth_raw - depth_raw.view(1,1,-1).mean(2).view(1,1,1,1)
            depth = depth.tanh()
            depth = self.depth_rescaler(depth)
            loss = F.mse_loss(depth[0], ellipsoid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0 and self.rank == 0:
                print(f"Iter: {i}, Loss: {loss.item():.6f}")

    def init_ellipsoid(self):
        with torch.no_grad():
            h, w = self.image_size, self.image_size
            c_x, c_y = w / 2, h / 2

            mask = self.input_mask[0, 0] >= 0.7
            max_y, min_y, max_x, min_x = utils.get_mask_range(mask)
            if self.category == 'synface':
                min_y = min_y - (max_y-min_y) / 6
            elif self.category == 'face':  # celeba
                max_y = h - 1
                width = max_x - min_x
                max_x -= width / 12
                min_x += width / 12
            elif self.category in ['car', 'church']:
                max_y = max_y + (max_y - min_y) / 6
            r_pixel = (max_x - min_x) / 2
            ratio = (max_y - min_y) / (max_x - min_x)
            c_x = (max_x + min_x) / 2
            c_y = (max_y + min_y) / 2
            radius = 0.4
            near = self.cfgs.get('prior_near', 0.91)
            far = self.cfgs.get('prior_far', 1.02)

            ellipsoid = torch.Tensor(1,h,w).fill_(far)
            i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))
            i = (i - h/2) / ratio + h/2
            temp = math.sqrt(radius**2 - (radius - (far - near))**2)
            dist = torch.sqrt((i - c_y)**2 + (j - c_x)**2)
            area = dist <= r_pixel
            dist_rescale = dist / r_pixel * temp
            depth = radius - torch.sqrt(torch.abs(radius ** 2 - dist_rescale ** 2)) + near
            ellipsoid[0, area] = depth[area]
            ellipsoid = ellipsoid.cuda()
            return ellipsoid

    def init_VL_sampler(self):
        from torch.distributions.multivariate_normal import MultivariateNormal as MVN
        view_mvn_path = self.cfgs.get('view_mvn_path', 'checkpoints/view_light/view_mvn.pth')
        light_mvn_path = self.cfgs.get('light_mvn_path', 'checkpoints/view_light/light_mvn.pth')
        view_mvn = torch.load(view_mvn_path)
        light_mvn = torch.load(light_mvn_path)
        self.view_mean = view_mvn['mean'].cuda()
        self.light_mean = light_mvn['mean'].cuda()
        self.view_mvn = MVN(view_mvn['mean'].cuda(), view_mvn['cov'].cuda())
        self.light_mvn = MVN(light_mvn['mean'].cuda(), light_mvn['cov'].cuda())

    def sample_view_light(self, num, sample_type='view'):
        samples = []
        for i in range(num):
            if sample_type == 'view':
                sample = self.view_mvn.sample()[None, :]
                sample[0, 1] *= self.view_scale
                samples.append(sample)
            else:
                samples.append(self.light_mvn.sample()[None, :])
        samples = torch.cat(samples, dim=0)
        return samples

    def init_parsing_model(self):
        if self.category in ['face', 'synface']:
            from .parsing import BiSeNet
            self.parse_model = BiSeNet(n_classes=19)
            self.parse_model.load_state_dict(torch.load('checkpoints/parsing/bisenet.pth', map_location=map_func))
        else:
            from .parsing import PSPNet
            if self.category == 'church':
                classes = 150
                ckpt_path = 'checkpoints/parsing/pspnet_ade20k.pth'
            else:
                classes = 21
                ckpt_path = 'checkpoints/parsing/pspnet_voc.pth'
            self.parse_model = PSPNet(classes=classes, pretrained=False)
            temp = nn.DataParallel(self.parse_model)
            checkpoint = torch.load(ckpt_path, map_location=map_func)
            temp.load_state_dict(checkpoint['state_dict'], strict=False)
            self.parse_model = temp.module

        self.parse_model = self.parse_model.cuda()
        self.parse_model.eval()

    def parse_mask(self, image):
        with torch.no_grad():
            size = 512 if self.category in ['face', 'synface'] else 473
            image = utils.resize(image, [size, size])
            if self.category in ['car', 'cat']:
                image = image / 2 + 0.5
                image[:, 0].sub_(0.485).div_(0.229)
                image[:, 1].sub_(0.456).div_(0.224)
                image[:, 2].sub_(0.406).div_(0.225)
            out = self.parse_model(image)
            if self.category in ['face', 'synface']:
                out = out[0]
            out = out.argmax(dim=1, keepdim=True)
            if self.category == 'face':
                mask_all = ((out >= 1) == (out != 16)).float()
                mask_face = ((out >= 1) == (out <= 13)).float()
                mask = (mask_all + mask_face) / 2
            elif self.category == 'synface':
                mask = ((out >= 1) == (out <= 14)).float()
            elif self.category == 'car':
                mask = (out == 7).float()
            elif self.category == 'cat':
                mask = (out == 8).float()
            elif self.category == 'church':
                mask = (out == 1).float()
            elif self.category == 'horse':
                mask = (out == 13).float()
        return utils.resize(mask, [self.image_size, self.image_size])

    def init_optimizers(self):
        self.optimizer_names = []
        if self.mode == 'step1':
            optimize_names = ['netA']
        elif self.mode == 'step2':
            optimize_names = ['netEnc']
        elif self.mode == 'step3':
            optimize_names = [name for name in self.network_names]
            optimize_names.remove('netEnc')

        for net_name in optimize_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                network = getattr(self, k)
                if hasattr(network, 'module'):
                    network.module.load_state_dict(cp[k])
                else:
                    network.load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            if self.distributed and self.share_weight:
                states[net_name] = getattr(self, net_name).module.state_dict()
            else:
                states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_total.backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward_step1(self):
        b = 1
        h, w = self.image_size, self.image_size
        ## predict depth
        self.depth_raw = self.netD(self.input_im).squeeze(1)  # 1xHxW
        self.depth = self.depth_raw - self.depth_raw.view(1,-1).mean(1).view(1,1,1)
        self.depth = self.depth.tanh()
        self.depth = self.depth_rescaler(self.depth)

        ## clamp border depth
        depth_border = torch.zeros(1,h,w-4).cuda()
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1.02)
        self.depth = self.depth*(1-depth_border) + depth_border *self.border_depth
        if (self.flip3 and self.mode == 'step3') or self.flip1:
            self.depth = torch.cat([self.depth, self.depth.flip(2)], 0)

        ## predict viewpoint transformation
        self.view = self.netV(self.input_im)
        if self.add_mean_V:
            self.view = self.view + self.view_mean.unsqueeze(0)
        if self.flip3 and self.mode == 'step3':
            self.view = self.view.repeat(2,1)
        self.view_trans = torch.cat([
            self.view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view[:,3:5] *self.xy_translation_range,
            self.view[:,5:] *self.z_translation_range], 1)
        self.renderer.set_transform_matrices(self.view_trans)

        ## predict albedo
        self.albedo = self.netA(self.input_im)  # 1x3xHxW
        if (self.flip3 and self.mode == 'step3') or self.flip1:
            self.albedo = torch.cat([self.albedo, self.albedo.flip(3)], 0)  # flip

        ## predict lighting
        self.light = self.netL(self.input_im)  # Bx4
        if self.add_mean_L:
            self.light = self.light + self.light_mean.unsqueeze(0)
        if (self.flip3 and self.mode == 'step3') or self.flip1:
            self.light = self.light.repeat(2,1)  # Bx4
        self.light_a = self.light[:,:1] /2+0.5  # ambience term
        self.light_b = self.light[:,1:2] /2+0.5  # diffuse term
        light_dxy = self.light[:,2:]
        self.light_d = torch.cat([light_dxy, torch.ones(self.light.size(0),1).cuda()], 1)
        self.light_d = self.light_d / ((self.light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        self.normal = self.renderer.get_normal_from_depth(self.depth)
        self.diffuse_shading = (self.normal * self.light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        self.shading = self.light_a.view(-1,1,1,1) + self.light_b.view(-1,1,1,1)*self.diffuse_shading
        self.texture = (self.albedo/2+0.5) * self.shading *2-1

        self.recon_depth = self.renderer.warp_canon_depth(self.depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
        if self.use_mask:
            if self.canon_mask is None or self.mode == 'step3' or self.joint_train:
                self.grid_2d_forward = self.renderer.get_warped_2d_grid(self.depth)
                self.canon_mask = nn.functional.grid_sample(self.input_mask, self.grid_2d_forward[0,None], mode='bilinear')
        self.grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        if (self.flip3 and self.mode == 'step3') or self.flip1:
            recon_im_mask = recon_im_mask[:b] * recon_im_mask[b:]
            recon_im_mask = recon_im_mask.repeat(2,1,1)
        recon_im_mask = recon_im_mask.unsqueeze(1).detach()
        self.recon_im = nn.functional.grid_sample(self.texture, self.grid_2d_from_canon, mode='bilinear').clamp(min=-1, max=1)

        ## loss function
        self.loss_l1_im = utils.photometric_loss(self.recon_im[:b], self.input_im, mask=recon_im_mask[:b])
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b] * recon_im_mask[:b], self.input_im * recon_im_mask[:b])
        self.loss_perc_im = torch.mean(self.loss_perc_im)
        if (self.flip3 and self.mode == 'step3') or self.flip1:
            self.loss_l1_im_flip = utils.photometric_loss(self.recon_im[b:], self.input_im, mask=recon_im_mask[b:])
            self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:]*recon_im_mask[b:], self.input_im*recon_im_mask[b:])
            self.loss_perc_im_flip = torch.mean(self.loss_perc_im_flip)

        self.loss_smooth = utils.smooth_loss(self.depth) + \
            utils.smooth_loss(self.diffuse_shading)
        self.loss_total = self.loss_l1_im + self.lam_perc*self.loss_perc_im + self.lam_smooth*self.loss_smooth
        if (self.flip3 and self.mode == 'step3') or self.flip1:
            self.loss_total += self.loss_l1_im_flip + self.lam_perc*self.loss_perc_im_flip

        metrics = {'loss': self.loss_total}

        ## compute accuracy if gt depth is available
        if hasattr(self, 'depth_gt'):
            self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)
            # mask out background
            mask_gt = (self.depth_gt<self.depth_gt.max()).float()
            mask_gt = (F.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask_pred = (F.avg_pool2d(recon_im_mask[:b], 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred
            self.acc_mae_masked = ((self.recon_depth[:b] - self.depth_gt[:b]).abs() *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.acc_mse_masked = (((self.recon_depth[:b] - self.depth_gt[:b])**2) *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth[:b].log(), self.depth_gt[:b].log(), mask=mask)
            self.acc_sie_masked = (self.sie_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1))**0.5
            self.norm_err_map_masked = utils.compute_angular_distance(self.recon_normal[:b], self.normal_gt[:b], mask=mask)
            self.acc_normal_masked = self.norm_err_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)

            metrics['SIDE'] = self.acc_sie_masked.mean()
            metrics['MAD'] = self.acc_normal_masked.mean()

        return metrics

    def step1_collect(self):
        # collect results in step1, used for step2 in joint training mode
        self.depth_all, self.albedo_all = [], []
        self.light_all, self.normal_all = [], []
        self.canon_mask_all = []
        self.img_idx = -1
        self.idx_perm = torch.LongTensor(list(range(self.img_num)))
        if self.rank == 0:
            print("Collecting step 1 results ...")
        for i in range(self.img_num):
            self.next_image()
            with torch.no_grad():
                self.forward_step1()
            self.depth_all.append(self.depth.cpu())
            self.albedo_all.append(self.albedo.cpu())
            self.light_all.append(self.light.cpu())
            self.normal_all.append(self.normal.cpu())
            if self.use_mask:
                self.canon_mask_all.append(self.canon_mask.cpu())

    def latent_project(self, image):
        offset = self.netEnc(image)
        if self.relative_enc:
            offset = offset - self.netEnc(self.gan_im)
        hidden = offset + self.center_h
        offset = self.generator.style_forward(hidden, skip=8-self.F1_d) - self.center_w
        latent = self.latent_w + offset
        return offset, latent

    def gan_invert(self, image, batchify=0):
        def gan_invert_sub(image):
            offset, latent = self.latent_project(image)
            gan_im, _ = self.generator([latent], input_is_w=True, truncation_latent=self.mean_latent,
                                       truncation=self.truncation, randomize_noise=False)
            return gan_im.clamp(min=-1, max=1), offset

        if batchify > 0:
            gan_ims, offsets = [], []
            for i in range(0, image.size(0), batchify):
                gan_im, offset = gan_invert_sub(image[i:i+batchify])
                gan_ims.append(gan_im)
                offsets.append(offset)
            gan_ims = torch.cat(gan_ims, dim=0)
            offsets = torch.cat(offsets, dim=0)
        else:
            gan_ims, offsets = gan_invert_sub(image)
        return gan_ims, offsets

    def forward_step2(self):
        with torch.no_grad():
            self.pseudo_im, self.mask = self.sample_pseudo_imgs(self.batchsize)

        self.proj_im, offset = self.gan_invert(self.pseudo_im)
        if self.crop is not None:
            self.proj_im = utils.resize(self.proj_im, [self.origin_size, self.origin_size])
            self.proj_im = utils.crop(self.proj_im, self.crop)
        self.proj_im = utils.resize(self.proj_im, [self.image_size, self.image_size])

        self.loss_l1 = utils.photometric_loss(self.proj_im, self.pseudo_im, mask=self.mask)
        self.loss_rec = self.d_loss(self.discriminator, self.proj_im, self.pseudo_im, mask=self.mask)
        self.loss_latent_norm = torch.mean(offset ** 2)
        self.loss_total = self.loss_l1 + self.loss_rec + self.lam_regular * self.loss_latent_norm

        metrics = {'loss': self.loss_total}
        return metrics

    def step2_collect(self):
        # collect projected samples, used for step3
        if self.joint_train:
            self.proj_im_all = [[] for i in range(self.img_num)]
            self.mask_all = [[] for i in range(self.img_num)]
        else:
            self.proj_imgs = []
            self.masks = []

        for i in range(self.collect_iters):
            if self.rank == 0 and i % 100 == 0:
                print(f"Collecting {i}/{self.collect_iters} samples ...")
            with torch.no_grad():
                self.forward_step2()
            if self.joint_train:
                idx = self.idx_perm[self.img_idx].item()
                self.proj_im_all[idx].append(self.proj_im.cpu())
                self.mask_all[idx].append(self.mask.cpu())
                self.next_image()
            else:
                self.proj_imgs.append(self.proj_im.cpu())
                self.masks.append(self.mask.cpu())

        if self.joint_train:
            for i in range(self.img_num):
                self.proj_im_all[i] = torch.cat(self.proj_im_all[i], 0)
                self.mask_all[i] = torch.cat(self.mask_all[i], 0)
        else:
            self.proj_imgs = torch.cat(self.proj_imgs, 0)
            self.masks = torch.cat(self.masks, 0)

    def forward_step3(self):
        # also reconstruct the input image using forward_step1
        metrics = self.forward_step1()
        # load collected projected samples
        self.load_proj_images(self.batchsize)
        b, h, w = self.proj_im.size(0), self.image_size, self.image_size

        ## predict viewpoint transformation
        view = self.netV(self.proj_im)
        if self.add_mean_V:
            view = view + self.view_mean.unsqueeze(0)
        view_trans = torch.cat([
            view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            view[:,3:5] *self.xy_translation_range,
            view[:,5:] *self.z_translation_range], 1)

        if self.flip3:
            view_trans = view_trans.repeat(2,1)
        self.renderer.set_transform_matrices(view_trans)

        ## predict lighting
        light = self.netL(self.proj_im)
        if self.add_mean_L:
            light = light + self.light_mean.unsqueeze(0)
        if self.flip3:
            light = light.repeat(2,1)  # Bx4
        light_a = light[:,:1] /2+0.5  # ambience term
        light_b = light[:,1:2] /2+0.5  # diffuse term
        light_dxy = light[:,2:]
        light_d = torch.cat([light_dxy, torch.ones(light.size(0),1).cuda()], 1)
        light_d = light_d / ((light_d**2).sum(1, keepdim=True))**0.5
        ## shading
        if self.flip3:
            self.normal = torch.cat([self.normal[0,None].repeat(b,1,1,1), self.normal[1,None].repeat(b,1,1,1)], 0)
        diffuse_shading = (self.normal * light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        shading = light_a.view(-1,1,1,1) + light_b.view(-1,1,1,1)*diffuse_shading

        if self.flip3:
            self.albedo = torch.cat([self.albedo[0,None].repeat(b,1,1,1), self.albedo[1,None].repeat(b,1,1,1)], 0)
        texture = (self.albedo/2+0.5) * shading *2-1

        if self.flip3:
            self.depth = self.depth[0,None,...].expand(b,h,w)
            self.depth = torch.cat([self.depth, self.depth.flip(2)], 0)
        else:
            self.depth = self.depth.expand(b,h,w)
        self.recon_depth = self.renderer.warp_canon_depth(self.depth)
        self.grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        if self.flip3:
            recon_im_mask = recon_im_mask[:b] * recon_im_mask[b:]
            recon_im_mask = recon_im_mask.repeat(2,1,1)
            mask = self.mask.repeat(2,1,1,1)
        else:
            mask = self.mask
        recon_im_mask = recon_im_mask.unsqueeze(1).detach() * mask
        self.recon_im = nn.functional.grid_sample(texture, self.grid_2d_from_canon, mode='bilinear').clamp(min=-1, max=1)

        ## loss function
        self.loss_l1_im = utils.photometric_loss(self.recon_im[:b], self.proj_im, mask=recon_im_mask[:b])
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b]*recon_im_mask[:b], self.proj_im*recon_im_mask[:b])
        self.loss_perc_im = torch.mean(self.loss_perc_im)
        if self.flip3:
            self.loss_l1_im_flip = utils.photometric_loss(self.recon_im[b:], self.proj_im, mask=recon_im_mask[b:])
            self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:]*recon_im_mask[b:], self.proj_im*recon_im_mask[b:])
            self.loss_perc_im_flip = torch.mean(self.loss_perc_im_flip)

        self.loss_total += self.loss_l1_im + self.lam_perc*self.loss_perc_im
        if self.flip3:
            self.loss_total += self.loss_l1_im_flip + self.lam_perc*self.loss_perc_im_flip

        metrics['loss'] = self.loss_total

        return metrics

    def forward(self):
        if self.mode == 'step1':
            m = self.forward_step1()
        elif self.mode == 'step2':
            m = self.forward_step2()
        elif self.mode == 'step3':
            m = self.forward_step3()
        return m

    def sample_pseudo_imgs(self, batchsize):
        b, h, w = batchsize, self.image_size, self.image_size

        # random lighting conditions
        # here we do not use self.sample_view_light, but use uniform distributions instead
        x_min, x_max, y_min, y_max, diffuse_min, diffuse_max, alpha = self.rand_light
        rand_light_dxy = torch.FloatTensor(b,2).cuda()
        rand_light_dxy[:,0].uniform_(x_min, x_max)
        rand_light_dxy[:,1].uniform_(y_min, y_max)
        rand_light_d = torch.cat([rand_light_dxy, torch.ones(b,1).cuda()], 1)
        rand_light_d = rand_light_d / ((rand_light_d**2).sum(1, keepdim=True))**0.5
        rand_diffuse_shading = (self.normal[0,None] * rand_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        rand = torch.FloatTensor(b,1,1,1).cuda().uniform_(diffuse_min, diffuse_max)
        rand_diffuse = (self.light_b[0,None].view(-1,1,1,1) + rand) * rand_diffuse_shading
        rand_shading = self.light_a[0,None].view(-1,1,1,1) + alpha * rand + rand_diffuse
        rand_light_im = (self.albedo[0,None]/2+0.5) * rand_shading * 2 - 1

        depth = self.depth[0,None]
        if self.use_mask:
            mask = self.canon_mask.expand(b,3,h,w)
        else:
            mask = torch.ones(b,3,h,w).cuda()

        # random viewpoints
        rand_views = self.sample_view_light(b, 'view')
        rand_views_trans = torch.cat([
            rand_views[:,:3] *math.pi/180 *self.xyz_rotation_range,
            rand_views[:,3:5] *self.xy_translation_range,
            rand_views[:,5:] *self.z_translation_range], 1)
        pseudo_im, mask = self.renderer.render_given_view(rand_light_im, depth.expand(b,h,w),
                                                          view=rand_views_trans, mask=mask, grid_sample=True)
        pseudo_im, mask = pseudo_im, mask[:,0,None,...]
        return pseudo_im.clamp(min=-1,max=1), mask.contiguous()

    def load_proj_images(self, batchsize):
        b = batchsize
        if self.joint_train:
            idx = self.idx_perm[self.img_idx].item()
            self.proj_imgs = self.proj_im_all[idx]
            self.masks = self.mask_all[idx]
            perm = torch.randperm(self.proj_imgs.size(0))
            rand_idx = perm[:b]
            self.proj_im = self.proj_imgs[rand_idx].cuda()
            self.mask = self.masks[rand_idx].cuda()
        else:
            self.proj_idx += b
            if self.proj_idx + b >= self.proj_imgs.size(0):
                self.proj_idx = 0
                rand_idx = torch.randperm(self.proj_imgs.size(0))
                self.proj_imgs = self.proj_imgs[rand_idx]
                self.masks = self.masks[rand_idx]
            self.proj_im = self.proj_imgs[self.proj_idx:self.proj_idx+b].cuda()
            self.mask = self.masks[self.proj_idx:self.proj_idx+b].cuda()

    def save_results(self, stage=0):
        path = self.image_path
        idx1 = path.rfind('/')
        idx2 = path.rfind('.')
        img_name = path[idx1+1:idx2]
        root = f'{self.checkpoint_dir}/images/{img_name}'
        if not os.path.exists(root):
            os.makedirs(root)
        flip = self.cfgs.get('flip3_cfg')[0]
        num_stage = self.cfgs.get('num_stage')

        def save_img(imgs, root, prefix, crop=False, last_only=False):
            if last_only and stage < num_stage:
                return
            if crop and self.crop is not None:
                imgs = utils.resize(imgs, [self.origin_size, self.origin_size])
                imgs = utils.crop(imgs, self.crop)
            if imgs.size(2) < 128:
                imgs = utils.resize(imgs, [128,128])
            for i in range(imgs.size(0)):
                save_path = f'{root}/{img_name}_{prefix}_stage{stage}_{i:03}.png'
                torchvision.utils.save_image(imgs[i,None], save_path, nrow=1)

        with torch.no_grad():
            depth, texture, view = self.depth[0,None], self.texture[0,None], self.view_trans[0,None]
            num_p, num_y = 5, 9  # number of pitch and yaw angles to sample
            max_y = 45 if self.category in ['car', 'church', 'horse'] else 70
            maxr = [20, max_y]
            # sample viewpoints
            im_rotate = self.renderer.render_view(texture, depth, maxr=maxr, nsample=[num_p,num_y])[0]
            save_img(im_rotate/2+0.5, root, 'im_rotate', last_only=True)
            im_rotate = self.renderer.render_view(texture, depth, maxr=maxr, nsample=[num_p,num_y], grid_sample=True)[0]
            gan_rotate, _ = self.gan_invert(im_rotate, batchify=10)
            save_img(gan_rotate/2+0.5, root, 'gan_rotate', crop=True, last_only=True)
            # sample relighting
            dxs = [-1.5, -0.7, 0, 0.7, 1.5]
            dys = [0, 0.6]
            rands = [0.3, 0.6]
            im_relightgs = []
            gan_relights = []
            for rand in rands:
                for dy in dys:
                    for dx in dxs:
                        light_dxy = torch.FloatTensor(1,2).cuda()
                        light_dxy[:,0].fill_(dx)
                        light_dxy[:,1].fill_(dy)
                        light_d = torch.cat([light_dxy, torch.ones(1,1).cuda()], 1)
                        light_d = light_d / ((light_d**2).sum(1, keepdim=True))**0.5
                        diffuse_shading = (self.normal[0,None] * light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
                        rand_diffuse = (self.light_b[0,None].view(-1,1,1,1) + rand) * diffuse_shading
                        rand_shading = self.light_a[0,None].view(-1,1,1,1) - 0.6 * rand + rand_diffuse
                        rand_texture = (self.albedo[0,None]/2+0.5) * rand_shading *2-1
                        im_relight = self.renderer.render_given_view(rand_texture, depth,
                                                                     view=view, grid_sample=False)
                        im_relightgs.append(im_relight.cpu())
                        im_relight = self.renderer.render_given_view(rand_texture, self.depth[0,None],
                                                                     view=view, grid_sample=True)
                        gan_relight, _ = self.gan_invert(im_relight, batchify=10)
                        gan_relights.append(gan_relight.cpu())
            im_relightgs = torch.cat(im_relightgs, dim=0)
            save_img(im_relightgs/2+0.5, root, 'im_relight', last_only=True)
            gan_relights = torch.cat(gan_relights, dim=0)
            save_img(gan_relights/2+0.5, root, 'gan_relight', crop=True, last_only=True)

            if flip:
                im_rotate = self.renderer.render_view(texture, depth, v_before=-view, maxr=maxr, nsample=[num_p,num_y])[0]
                save_img(im_rotate/2+0.5, root, 'im_rotate2', last_only=True)
            if self.use_mask:
                mask = self.canon_mask.clone()
                mask[mask<0.3] = 0
                mask[mask>=0.3] = 1
                im_masked = (texture/2+0.5 - 1) * mask + 1
                im_masked_rotate = self.renderer.render_view(im_masked, depth, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
                save_img(im_masked_rotate, root, 'im_masked_rotate', last_only=True)
                im_masked_relight = (im_relightgs/2+0.5 - 1) * mask.cpu() + 1
                save_img(im_masked_relight, root, 'im_masked_relight', last_only=True)
            # render normal and shape
            normal = self.renderer.get_normal_from_depth(depth)
            front_light = torch.FloatTensor([0,0,1]).cuda()
            shape_im = (normal * front_light.view(1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
            shape_im = shape_im.repeat(1,3,1,1) * 0.7
            normal = normal[0,None].permute(0,3,1,2)/2+0.5
            normal_rotate = self.renderer.render_view(normal, depth, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
            shape_rotate = self.renderer.render_view(shape_im, depth, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
            save_img(normal_rotate, root, 'normal_rotate')
            save_img(shape_rotate, root, 'shape_rotate')
            if flip:
                normal_rotate = self.renderer.render_view(normal, depth, v_before=-view, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
                shape_rotate = self.renderer.render_view(shape_im, depth, v_before=-view, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
                save_img(normal_rotate, root, 'normal_rotate2', last_only=True)
                save_img(shape_rotate, root, 'shape_rotate2', last_only=True)
            if self.use_mask:
                normal = (normal - 1) * mask + 1
                shape_im = (shape_im - 1) * mask + 1
                normal_rotate = self.renderer.render_view(normal, depth, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
                shape_rotate = self.renderer.render_view(shape_im, depth, maxr=maxr, nsample=[num_p,num_y])[0].cpu()
                save_img(normal_rotate, root, 'normal_masked_rotate')
                save_img(shape_rotate, root, 'shape_masked_rotate')
            # save albedo
            save_img(self.albedo[0,None]/2+0.5, root, 'albedo')
            if flip:
                albedo2 = nn.functional.grid_sample(self.albedo[0,None], self.grid_2d_from_canon[0,None], mode='bilinear').clamp(min=-1, max=1)
                albedo2[albedo2==0] = 1
                save_img(albedo2/2+0.5, root, 'albedo2', last_only=True)
            # save mesh
            vertices = self.depth_to_3d_grid(depth)  # BxHxWx3
            normal = self.renderer.get_normal_from_depth(depth)
            self.objs, self.mtls = utils.export_to_obj_string(vertices, normal)
            torchvision.utils.save_image(texture/2+0.5, f'{root}/{img_name}_canonical_image.png', nrow=1)
            with open(os.path.join(root, f'{img_name}_mesh.mtl'), "w") as f:
                f.write(self.mtls[0].replace('$TXTFILE', f'./{img_name}_canonical_image.png'))
            with open(os.path.join(root, f'{img_name}_mesh.obj'), "w") as f:
                f.write(self.objs[0].replace('$MTLFILE', f'./{img_name}_mesh.mtl'))

    def depth_to_3d_grid(self, depth, inv_K=None):
        if inv_K is None:
            inv_K = self.renderer.inv_K
        b, h, w = depth.shape
        grid_2d = utils.get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(inv_K.transpose(2,1)) * depth
        return grid_3d

    def visualize_results(self, logger, iteration):
        b = 1 if self.mode == 'step1' else self.batchsize
        depth, texture, view = self.depth[0,None], self.texture[0,None], self.view_trans[0,None]

        def log_grid_image(label, im, iteration=iteration):
            nrow=int(math.ceil(im.size(0)**0.5))
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iteration)

        if self.rank == 0:
            with torch.no_grad():
                im_rotate = self.renderer.render_yaw(texture, depth, v_before=-view, maxr=90)[0].cpu() /2+0.5

        if self.distributed:
            recon_imgs = [self.recon_im[:b].clone().zero_() for i in range(dist.get_world_size())]
            input_imgs = [self.input_im.clone().zero_() for i in range(dist.get_world_size())]
            dist.all_gather(recon_imgs, self.recon_im[:b])
            dist.all_gather(input_imgs, self.input_im)
            recon_imgs = torch.cat(recon_imgs, dim=0)
            input_imgs = torch.cat(input_imgs, dim=0)
        else:
            recon_imgs = self.recon_im[:b]
            input_imgs = self.input_im

        ## write summary
        if self.rank == 0:
            logger.add_scalar('Loss/loss_total', self.loss_total, iteration)
            logger.add_scalar('Loss/loss_l1_im', self.loss_l1_im, iteration)
            logger.add_scalar('Loss/loss_perc_im', self.loss_perc_im, iteration)

            log_grid_image('Image/recon_images', recon_imgs/2+0.5)
            log_grid_image('Image/im_rotate', im_rotate)
            logger.add_image('Image/depth', self.depth_inv_rescaler(depth), iteration)
            logger.add_image('Image/albedo', self.albedo[0,...]/2+0.5, iteration)
            with torch.no_grad():
                # render normal and shape
                normal = self.renderer.get_normal_from_depth(depth)
                front_light = torch.FloatTensor([0,0,1]).cuda()
                shape_im = (normal * front_light.view(1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
                shape_im = shape_im.repeat(1,3,1,1) *0.7
                normal = normal[0].permute(2,0,1)/2+0.5
            logger.add_image('Image/normal', normal, iteration)
            logger.add_image('Image/shape', shape_im[0], iteration)
            with torch.no_grad():
                normal_rotate = self.renderer.render_yaw(normal.unsqueeze(0), depth, v_before=-view, maxr=90)[0].cpu()
                shape_rotate = self.renderer.render_yaw(shape_im, depth, v_before=-view, maxr=90)[0].cpu()
            log_grid_image('Image/normal_rotate', normal_rotate)
            log_grid_image('Image/shape_rotate', shape_rotate)
            log_grid_image('Image/input_im', input_imgs/2+0.5)

        if hasattr(self, 'depth_gt'):
            depth_gt = ((self.depth_gt -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0,3,1,2).detach().cpu() /2+0.5
            sie_map_masked = self.sie_map_masked.detach().unsqueeze(1).cpu() *1000
            norm_err_map_masked = self.norm_err_map_masked.detach().unsqueeze(1).cpu() /100

            acc_mae_masked = self.acc_mae_masked.mean()
            acc_mse_masked = self.acc_mse_masked.mean()
            acc_sie_masked = self.acc_sie_masked.mean()
            acc_normal_masked = self.acc_normal_masked.mean()

            if self.distributed:
                dist.all_reduce(acc_mae_masked)
                dist.all_reduce(acc_mse_masked)
                dist.all_reduce(acc_sie_masked)
                dist.all_reduce(acc_normal_masked)
                acc_mae_masked /= dist.get_world_size()
                acc_mse_masked /= dist.get_world_size()
                acc_sie_masked /= dist.get_world_size()
                acc_normal_masked /= dist.get_world_size()

            if self.rank == 0:
                logger.add_scalar('Acc_masked/MAE_masked', acc_mae_masked, iteration)
                logger.add_scalar('Acc_masked/MSE_masked', acc_mse_masked, iteration)
                logger.add_scalar('Acc_masked/SIE_masked', acc_sie_masked, iteration)
                logger.add_scalar('Acc_masked/NorErr_masked', acc_normal_masked, iteration)

                log_grid_image('Depth_gt/depth_gt', depth_gt)
                log_grid_image('Depth_gt/normal_gt', normal_gt)
                log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
                log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)

    def visualize_pseudo_proj(self, logger, iteration):
        def log_grid_image(label, im, iteration=iteration):
            nrow=int(math.ceil(im.size(0)**0.5))
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iteration)

        if self.distributed:
            if hasattr(self, 'pseudo_im') and self.pseudo_im is not None:
                pseudo_imgs = [self.pseudo_im.clone().zero_() for i in range(dist.get_world_size())]
                dist.all_gather(pseudo_imgs, self.pseudo_im)
                pseudo_imgs = torch.cat(pseudo_imgs, dim=0)
            proj_imgs = [self.proj_im.clone().zero_() for i in range(dist.get_world_size())]
            masks = [self.mask.clone().zero_() for i in range(dist.get_world_size())]
            dist.all_gather(proj_imgs, self.proj_im)
            dist.all_gather(masks, self.mask)
            proj_imgs = torch.cat(proj_imgs, dim=0)
            masks = torch.cat(masks, dim=0)
        else:
            if hasattr(self, 'pseudo_im') and self.pseudo_im is not None:
                pseudo_imgs = self.pseudo_im
            proj_imgs = self.proj_im
            masks = self.mask

        ## write summary
        if self.rank == 0:
            if self.mode == 'step2':
                log_grid_image('Image/pseudo_images', pseudo_imgs/2+0.5, iteration)
            log_grid_image('Image/proj_images', proj_imgs/2+0.5, iteration)
            log_grid_image('Image/mask', masks, iteration)

    def visualize(self, logger, iteration):
        if self.mode in ['step2', 'step3']:
            self.visualize_pseudo_proj(logger, iteration)
        if self.mode in ['step1', 'step3']:
            self.visualize_results(logger, iteration)
