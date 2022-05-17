import numpy as np
import os
import pandas as pd
import argparse
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from label_selection import choose_target
from sample_generation import attack_to_target
from util import *

################################
# Preparation
################################

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# prepossess image
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])


# Attack images in ./Original
df = pd.read_csv('./Original/groundtruth.csv')
files = df['filename'].values
truth = df['groundtruth'].values


def main(args):
    # load models
    net = load_model(args.model)
    net.eval()

    # read image and its ground truth
    im_orig = Image.open(os.path.join(args.img_dir,args.filename))
    im = trans(im_orig)

    # label selection
    labelfool = choose_target(im, net, args.matrix_path, args.target_path)

    # corresponding words for labels
    labels = open(os.path.join('./Original/synset_words.txt'), 'r').read().split('\n')
    str_label_orig = labels[np.int(args.org_label)].split(',')[0]
    str_label_pert = labels[np.int(labelfool)].split(',')[0]
    print("The groundtruth of image {} is {}, LabelFool choose {} as the target label.".format(args.filename, str_label_orig,
                                                                                               str_label_pert))

    # sample generation
    r, loop_i, pert_image, flag = attack_to_target(im, net, labelfool)

    # save perturbed images
    save_res(args.save_dir, args.filename, pert_image)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--model',help='target model: alexnet, vgg16, resnet50', type=str, default='resnet50')
    parser.add_argument('--filename',help='path to testing image', type=str, required=True)
    parser.add_argument('--org_label',help='ground true label of testing image', type=int, required=True)
    parser.add_argument('--img_dir', help='original images', type=str, default='./Original')
    parser.add_argument('--save_dir', help='results saving directory', type=str, default='./Perturbed Images')
    parser.add_argument('--matrix_path', help='Matrix D', type=str, default='./ImageNet_Matrix.npy')
    parser.add_argument('--target_path', help='target[i] records the nearest label to i according to Matrix D', type=str, default='./ImageNet_target.npy')
    args = parser.parse_args()

    main(args)
