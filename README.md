# Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs

<p align="center">
    <img src="GAN2Shape_demo.gif", width="900">
</p>

**Figure:** *Recovered 3D shape and rotation&relighting effects using GAN2Shape.*

> **Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs** <br>
> Xingang Pan, Bo Dai, Ziwei Liu, Chen Change Loy, Ping Luo <br>
> *ICLR2021* (**Oral**)

[[Paper](https://openreview.net/pdf?id=FGqiDsBUKL0)]
[[Project Page](https://xingangpan.github.io/projects/GAN2Shape.html)]

In this repository, we present **GAN2Shape**, which reconstructs the 3D shape of an image using off-the-shelf 2D image GANs in an unsupervised manner.
Our method **does not rely on mannual annotations or external 3D models**, yet it achieves high-quality 3D reconstruction, object rotation, and relighting effects.

## Requirements

* python>=3.6
* [pytorch](https://pytorch.org/)=1.1 or 1.2
* [neural_renderer](https://github.com/daniilidis-group/neural_renderer)
    ```sh
    pip install neural_renderer_pytorch  # or follow the guidance at https://github.com/elliottwu/unsup3d
    ```
* [mmcv](https://github.com/open-mmlab/mmcv)
    ```sh
    pip install mmcv
    ```
* other dependencies
    ```sh
    conda install -c conda-forge scikit-image matplotlib opencv pyyaml tensorboardX
    ```

## Dataset and pre-trained weights

To download dataset and pre-trained weights, simply run:
```sh
sh scripts/download.sh
```

## Training

Before training, you may optionally compile StyleGAN2 operations, which would be faster:
```sh
cd gan2shape/stylegan/stylegan2-pytorch/op
python setup.py install
cd ../../../..
```

**Example1**: training on car images:
```sh
sh scripts/run_car.sh
```
You can view the results at `results/car/images` or Tensorboard.

**Example2**: training on Celeba images:
```sh
sh scripts/run_celeba.sh
```
This by default uses our provided pre-trained weights. You can also perform joint pre-training via:
```sh
sh scripts/run_celeba-pre.sh
```

**Example3**: evaluating on synface (BFM) dataset:
```sh
sh scripts/run_synface.sh
```
This by default uses our provided pre-trained weights. You can also perform joint pre-training via:
```sh
sh scripts/run_synface-pre.sh
```

If you want to train on new StyleGAN2 samples, simply run the following script to generate new samples:
```sh
sh scripts/run_sample.sh
```

**Note**:  
\- For human and cat faces, we perform joint training before instance-specific training, which produces better results.  
\- For car and church, the quality of StyleGAN2 samples vary a lot, thus our approach may not produce good result on every sample. The downloaded dataset contains examples of good samples.

## Acknowledgement

Part of the code is borrowed from [Unsup3d](https://github.com/elliottwu/unsup3d) and [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).

## BibTeX

```bibtex
@inproceedings{pan2020gan2shape,
  title   = {Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs},
  author  =  {Pan, Xingang and Dai, Bo and Liu, Ziwei and Loy, Chen Change and Luo, Ping},
  booktitle = {ICLR},
  year    = {2021}
}
```
