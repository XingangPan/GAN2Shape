import os
import argparse

import torch
import torch.nn.functional as F
from torchvision import utils
from model import Generator
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        count = 0
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)
           sample_w = g_ema.style_forward(sample_z)

           sample, _ = g_ema([sample_w], truncation=args.truncation, truncation_latent=mean_latent, input_is_w=True)
           sample_w = mean_latent + args.truncation * (sample_w - mean_latent)
           
           for j in range(args.sample):
                utils.save_image(
                    sample[j],
                    f'{args.save_path}/{str(count).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                torch.save(sample_w[j], f'{args.save_path}/latents/{str(count).zfill(6)}.pt')
                count += 1

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='sample')

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    torch.manual_seed(args.seed) # also sets cuda seeds

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(args.save_path + '/latents')

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
