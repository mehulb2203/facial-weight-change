import numpy as np
import pandas as pd
import os
import argparse
import cv2
from skimage.measure import compare_ssim as ssim


def main():
    parser = argparse.ArgumentParser(description='Calculate SSIM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Directory with source images')
    parser.add_argument('generated_images_dir', help='Directory with generated images')

    args, other_args = parser.parse_known_args()

    ref_images_paths = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    generated_images_paths = [os.path.join(args.generated_images_dir, x) for x in os.listdir(args.generated_images_dir)]
    ref_images_paths = sorted(list(filter(os.path.isfile, ref_images_paths)))
    generated_images_paths = sorted(list(filter(os.path.isfile, generated_images_paths)))

    ssims = []
    for idx, ref_images_path in enumerate(ref_images_paths):
        name = (os.path.splitext(os.path.basename(ref_images_path))[0]).split('/')[-1]
        # Read images
        original = cv2.imread(ref_images_paths[idx])
        embedded = cv2.imread(generated_images_paths[idx], 1)

        ssim_score = ssim(original, embedded, multichannel=True)
        ssims.append(round(ssim_score, 2))
        print("SSIM score for ", name, " is:\t", round(ssim_score, 2), "\n")
    #ssims_sr = pd.Series(ssims)
    #summary = ssims_sr.describe()



if __name__ == '__main__':
    main()