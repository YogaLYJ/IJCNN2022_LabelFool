# LabelFool

Code for 'LabelFool: A Trick In The Label Space' (IJCNN 2022).

[Paper (The proceedings of IJCNN2022 have not yet been published)](Place Hold).

## Recommended Environment
* Python 3.7
* Cuda 10.1
* PyTorch 0.4

## Prerequisites
* Install dependencies: `pip3 install -r requirements.txt`.

## Document Description

### ./Files
We use ResNet101 to extract features for all images in ImageNet training set.

* ImageNet\_Matrix.npy: matrix `$D$` for ImageNet in the paper. `$D[i,j]$` represents the Hausdorff distance between class `$i$` and class `$j$`.
Remark: To prevent `$\arg\min_{j\in\mathcal{L}} D[i,j] = i$`, we set the diagonal values to a large number, `$D[i,i]$=1000`.
* ImageNet\_target.npy: this is a 1000*1 vector for ImageNet. The `$i^{th}$` element represents the nearest label to `$i$` according to `$D[i,j]$`, i.e.
```math
\text{target}[i] = \arg\min_{
    j\in\mathcal{L}
} D[i,j]
```
where `$\mathcal{L}=\{1,2,\dots,1000\}$`.

### ./Original

* \*.JPEG: original Image from ImageNet validation set.
* groundtruth.csv: save the name of images and corresponding groundtruth (a number from `$\{1,2,\dots,1000\}$`).
* synset_words.txt

### Others
* label_selection.py: code for label selection part.
* sample_generation.py: code for sample generation part which is adapted from [DeepFool](https://github.com/LTS4/DeepFool/tree/master/Python).

## Quick Start

### Parameters
* model: the model to be attacked
* filename: the image to be attacked
* org_label: the groundtruth of the image
* img_dir: the directory where orginal images are stored
* save_dir: the directory where perturbed images will be stored
* matrix\_path: the path of ImageNet\_Matrix.npy
* target\_path: the path of ImageNet\_target.npy

### Run: attack an image with LabelFool
```
python main.py --model=resnet50 --filename=n01491361_10052.JPEG --org_label=3 --save_dir=./Perturbed Images
```

## Citation
@InProceedings{Liu_2022_IJCNN_LabelFool,
author = {Liu, Yujia and Jiang, Ming and Jiang, Tingting},
title = {{LabelFool}: A Trick In The Label Space},
booktitle = {International Joint Conference on Neural Networks (IJCNN)},
month = {July},
year = {2022}
}
