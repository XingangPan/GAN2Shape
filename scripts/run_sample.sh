# This script would sample 100(pics) x 10(sample) = 1000 images

python gan2shape/stylegan2/stylegan2-pytorch/generate.py \
    --pics 100 --sample 10 \
    --size 128 --channel_multiplier 1 \
    --ckpt checkpoints/stylegan2/stylegan2-celeba-config-e.pt \
    --save_path data/celeba_sample
