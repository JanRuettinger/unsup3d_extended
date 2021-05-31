

This repository contains a Pytorch3D port of the code of the paper "[Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/abs/1911.11130)" by [Wu](https://elliottwu.com) et al. You can find the original code [here](https://github.com/elliottwu/unsup3d).


## Differences compared to original version
- [Pytorch3D](https://github.com/facebookresearch/pytorch3d) instead of [Neural Renderer](https://github.com/hiroharu-kato/neural_renderer) is used as a differential renderer. 
- The size of the albedo map (new: 128x128) and depth map (new: 32x32) have changed. Gradient update is unstable with original size of 64x64 for both maps.

## Setup (with Anaconda)
```
conda env create -f environment.yml
```
Tested with Python 3.8.3.

## Datasets

1. CelebA face dataset. Please download the original images (img_celeba.7z) from their website and run celeba_crop.py in data/ to crop the images.
2. Cat face dataset composed of Cat Head Dataset and Oxford-IIIT Pet Dataset (license). This can be downloaded using the script download_cat.sh provided in data/.

Please remember to cite the corresponding papers if you use these datasets.

## Pretrained Models
Pretrained models are provided for both datasets in the folder `pretrained`. Set the config option `resume=True` and copy the pretrained model to your `checkpoint_dir`.

## Training and Testing
```
python run.py --config experiments/train_celeba.yml --gpu 0 --num_workers 4
```
Training & testing without a GPU is not recommended.

## Known warnings/issues:
1. `UserWarning: No mtl file provided`
Warning appears when a depth map prior is loaded. Material information is not needed here. Warning can be ignoered.

## Citations
```
@InProceedings{Wu_2020_CVPR,
  author = {Shangzhe Wu and Christian Rupprecht and Andrea Vedaldi},
  title = {Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild},
  booktitle = {CVPR},
  year = {2020}
}
```


