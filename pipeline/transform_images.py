"""
    Script to generate thinner and heavier facial weight transformations.
    Usage:  CUDA_VISIBLE_DEVICES=$DEVICE_ID$ python transform_images.py $PATH_TO_LATENT_CODES$
"""
import numpy as np
import os
import argparse
import pickle
import PIL.Image

import .dnnlib.tflib as tflib
from .encoder.generator_model import Generator


# Load StyleGAN-ffhq
tflib.init_tf()
with open('../cache/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
   generator_network, discriminator_network, Gs_network = pickle.load(f)

# Initialize the Synthesis network and load the 'weight' attribute direction
generator = Generator(Gs_network)
direction = np.load('../cache/weight_orth_mouth.npy')


def generate_image(latent_vector):
    '''
    Function for converting latent code to its image representation
    parameters: 18x512 dimensional embedded latent code
    output: image at 1024x1024 resolution
    '''
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


def move_and_save_ind(latent_vector, coeffs, filename='default'):
    '''
    Funtion that uses the embedded latent code, weight direction, and coefficients 
        to generate facial weight transformations
    parameters: 18x512 dimensional embedded latent code; coefficients for steering along in the latent space; optional folder name to store the 5 transformations
    output: Folder with 5 transformed images at 1024x1024 resolution
    '''
    dest_dir = 'transformed_images/'
    os.makedirs(dest_dir + filename, exist_ok=True)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        # Perform simple vector arithmetic using the first 8 layers of the latent code
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        img = generate_image(new_latent_vector)
        imagename = filename + '_' + str(coeff)
        path = os.path.join('../transformed_images', filename, f'{imagename}.png')
        img.save(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Facial Weight Transformations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Source directory with the embedded latent codes')

    args, _ = parser.parse_known_args()

    # Store the absolute paths to the latent codes in a sorted list
    latent_code_paths = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    latent_code_paths = sorted(list(filter(os.path.isfile, latent_code_paths)))

    # Iterate over all the latent codes and generate the corresponding thinnest (-5), thinner (-3), normal (0), heavier (+3), heaviest (+5) transformations
    for idx, latent_code_path in enumerate(latent_code_paths):
        latent_code = np.load(latent_code_paths[idx])
        name = (os.path.splitext(os.path.basename(latent_code_path))[0]).split('/')[-1]
        move_and_save_ind(latent_code, [-5, -3, 0, 3, 5], name)