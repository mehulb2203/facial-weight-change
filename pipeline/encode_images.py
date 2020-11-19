"""
    Main script pertaining to Step-2 (i.e. Latent Space Embedding) of the pipeline.
    Usage:  CUDA_VISIBLE_DEVICES=$DEVICE_ID$ python encode_images.py $PATH_TO_ALIGNED_IMAGES$ $PATH_FOR_GENERATED_IMAGES$ $PATH_FOR_LATENT_CODES$
"""
import numpy as np
import tensorflow as tf
import os
import time
import csv
import pickle
import argparse
import PIL.Image
import json
import requests
from io import BytesIO
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input
import keras.backend as K

import .dnnlib
import .dnnlib.tflib as tflib
from .encoder.generator_model import Generator
from .encoder.perceptual_model import load_images, PerceptualModel


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    # Initial global params
    parser = argparse.ArgumentParser(description='Find latent representation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Directory with the aligned images')
    parser.add_argument('generated_images_dir', help='Directory for storing the generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing the latent representations')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the ResNet model', type=int)
    parser.add_argument('--lr', default=0.01, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Loss function options
    parser.add_argument('--use_vgg_loss', default=1, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_pixel_loss', default=1, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

    args, _ = parser.parse_known_args()

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with tf.get_default_session() as sess:
        with open('../cache/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)

        generator = Generator(Gs_network)
        perceptual_model = PerceptualModel(args, batch_size=args.batch_size, sess=sess)

        generated_image = tf.image.resize_nearest_neighbor(generator.generated_image, (args.image_size, args.image_size), align_corners=True)

        resize_tensor = generated_image[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        _IMAGENET_MEAN = tf.constant(-np.array(mean), dtype=K.floatx())
        resize_tensor = generated_image = K.bias_add(resize_tensor, _IMAGENET_MEAN, 'channels_last')

        generated_img_features = perceptual_model.perceptual_model(resize_tensor)

        perceptual_model.ref_img = tf.get_variable('ref_img', shape=generated_image.shape, dtype='float32', initializer=tf.initializers.zeros())
        perceptual_model.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape, dtype='float32', initializer=tf.initializers.zeros())

        perceptual_model.calculate_loss(generated_image, generated_img_features)

        emb_info_list = []
        emb_info_list.append(["NAME", "TOTAL TIME", "LOSS"])
        # Optimize dlatents by minimizing perceptual loss between reference and generated images in feature space
        for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
            start = time.process_time()
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

            perceptual_model.set_reference_images(images_batch)

            dlatents = None
            ##############################   TF-Serving   ##############################
            pre_processed_data = load_images(images_batch, image_size=args.resnet_image_size)
            data = json.dumps({"signature_name": "serving_default", "instances": pre_processed_data.tolist()})
            headers = {"content-type": "application/json"}
            print("\nStarting ResNet prediction.....")
            json_response = requests.post('http://localhost:8501/v1/models/resnet:predict', data=data, headers=headers)
            print("\nPrediction SUCCESSFUL.....\n\n")
            dlatents = tf.convert_to_tensor(json.loads(json_response.text)['predictions'], dtype=tf.dtypes.float32)
            ##############################   TF-Serving   ##############################
            generator.set_dlatents(dlatents)

            print("Starting optimization...\n")
            op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, learning_rate=args.lr)
            pbar = tqdm(op, leave=False, total=args.iterations)
            best_loss = None
            best_dlatent = None
            for loss in pbar:
                # pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.2f}".format(k, v) for k, v in loss_dict.items()]))
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_dlatent = generator.get_dlatents()
                generator.stochastic_clip_dlatents()

            # Generate images from found dlatents and save them
            generator.set_dlatents(best_dlatent)
            generated_images = generator.generate_images()
            generated_dlatents = generator.get_dlatents()

            for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
                img = PIL.Image.fromarray(img_array, 'RGB')
                img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
                np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

            generator.reset_dlatents()

            total_time = time.process_time() - start

            total_time_min = int(total_time // 60)
            if total_time_min == 0:
                total_time_min = str(0) + str(total_time_min)

            total_time_sec = int(total_time % 60)
            if total_time_sec >= 0 and total_time_sec <= 9:
                total_time_sec = str(0) + str(total_time_sec)

            total_time_str = str(total_time_min) + ":" + str(total_time_sec)
            emb_info_list.append([names[0], total_time_str + "s", round(best_loss, 2)])

            # Save the info. regarding computation time, loss etc. in a CSV file
            with open('embedding_info.csv', mode='w') as emb_file:
                emb_writer = csv.writer(emb_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for lst in emb_info_list:
                    emb_writer.writerow(lst)



if __name__ == '__main__':
    main()