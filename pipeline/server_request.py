"""
    Usage: tensorflow_model_server --rest_api_port=8501 --model_name=resnet --model_base_path=../../models/ResNet/
"""
import tensorflow as tf
import os
import pickle
import PIL.Image
import json
import requests
from keras.applications.resnet50 import preprocess_input

import ..dnnlib
import ..dnnlib.tflib as tflib
from ..encoder.generator_model import Generator
from ..encoder.perceptual_model import load_images


images_path = '../../raw_images/'

# Initialize generator
tflib.init_tf()
with open('../../cache/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
   generator_network, discriminator_network, Gs_network = pickle.load(f)
generator = Generator(Gs_network)

images = [os.path.join(images_path, image) for image in os.listdir(images_path)]
pre_processed_data = preprocess_input(load_images(images, image_size=256))
data = json.dumps({"signature_name": "serving_default", "instances": pre_processed_data.tolist()})

print("Starting ResNet prediction.....")
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/resnet:predict', data=data, headers=headers)
print("Prediction SUCCESSFUL.....")

dlatents = tf.convert_to_tensor(json.loads(json_response.text)['predictions'], dtype=tf.dtypes.float32)
generator.set_dlatents(dlatents)

generated_images = generator.generate_images()
for generated_image in generated_images:
    img = PIL.Image.fromarray(generated_image, 'RGB')
    img.save('Sample.png', 'PNG')