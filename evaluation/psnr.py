# Importing all the necessary packages
import numpy as np
import pandas as pd
import os
import cv2
from math import log10, sqrt
import argparse


def PSNR(original, embedded):
    mse = np.mean((original - embedded) ** 2)
    if(mse == 0): # If MSE is zero, it means there is no noise present in the signal
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Directory with source images')
    parser.add_argument('generated_images_dir', help='Directory with generated images')

    args, other_args = parser.parse_known_args()

    ref_images_paths = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    generated_images_paths = [os.path.join(args.generated_images_dir, x) for x in os.listdir(args.generated_images_dir)]
    ref_images_paths = sorted(list(filter(os.path.isfile, ref_images_paths)))
    generated_images_paths = sorted(list(filter(os.path.isfile, generated_images_paths)))

    psnrs = []
    for idx, ref_images_path in enumerate(ref_images_paths):
        name = (os.path.splitext(os.path.basename(ref_images_path))[0]).split('/')[-1]
        # Read images
        original = cv2.imread(ref_images_paths[idx])
        embedded = cv2.imread(generated_images_paths[idx], 1)

        psnr = PSNR(original, embedded)
        psnrs.append(round(psnr, 2))
        print("PSNR value for ", name, " is:\t", round(psnr, 2), " dB\n")
    #psnrs_sr = pd.Series(psnrs)
    #summary = psnrs_sr.describe()



if __name__ == '__main__':
    main()