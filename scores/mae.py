import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from glob import glob
from ntpath import basename
from skimage.color import rgb2gray
from scipy.spatial import distance


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', '--gt', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', '--o', help='Path to output data', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path = args.data_path
output = args.output_path

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

def MAE(yhat, y):
    loss = np.mean(np.abs(yhat-y))/np.mean(yhat)
    return loss


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

# gt = load_flist(path)
gt = list(glob(path + '/*.jpg')) + list(glob(path + '/*.png'))+list(glob(path + '/*.JPG'))

mae = []
  

for fn in sorted(gt):
    name = basename(str(fn))
    pred_name = str(fn)
    img_gt = imread(str(fn))
    img_gt = img_resize(img_gt, 256, 256)
    img_out = imread(output + basename(pred_name)) 
    print(MAE(img_gt, img_out))
    mae.append(MAE(img_gt, img_out))

print(
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4),
)
