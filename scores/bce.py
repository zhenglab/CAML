import os
import numpy as np
import argparse
import cv2

from glob import glob
from ntpath import basename
from imageio import imread
# import tensorflow as tf
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', '--gt', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', '--o', help='Path to output data', type=str)
    parser.add_argument('--valrst-path', '--v', default='./', help='Path to save val result', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

def load_flist(flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))+list(glob(path_true + '/*.JPG'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

bces = []

names = []
index = 1
total_bce = 0
# files = load_flist(path_true)
files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))+list(glob(path_true + '/*.JPG'))
test_num = len(files)

def img_resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(height, width))

        return img
    else: 
        #img = cv2.resize(img, dsize=(height, width))

        return img
    
for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    pred_name = str(fn)
    
    img_gt = (imread(str(fn))/255.).astype(np.float32)
    img_pred = (imread(path_pred + '/' + basename(pred_name))/255.).astype(np.float32)
    # np.set_printoptions(threshold=np.inf)
    mask = torch.from_numpy(img_gt)
    mask_logit = torch.from_numpy(img_pred)
    bce_loss = reduce_mean(mask*-torch.log(torch.sigmoid(mask_logit)) + (1-mask)*-torch.log(1-torch.sigmoid(mask_logit)))
    index += 1
    total_bce += bce_loss.numpy()
    # print(total_bce)
    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "BCE: %.4f" % round(np.mean(total_bce/(index-1)), 4)
        )
print('BCE: %.4f'% round(np.mean(total_bce / test_num), 4))