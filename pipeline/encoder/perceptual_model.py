import numpy as np
import tensorflow as tf
import PIL.Image
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K


def load_images(images_list, image_size=256):
    loaded_images = list()
    for img_path in images_list:
        img = PIL.Image.open(img_path).convert('RGB').resize((image_size, image_size), PIL.Image.LANCZOS)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


def tf_custom_l1_loss(img1, img2):
  return tf.math.reduce_mean(tf.math.abs(img2 - img1), axis=None)


def tf_custom_logcosh_loss(img1, img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1, img2))


class PerceptualModel:
    def __init__(self, args, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.img_size = args.image_size

        self.vgg_loss = args.use_vgg_loss
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None

        self.batch_size = batch_size
        self.ref_img = None
        self.ref_img_features = None
        self.loss = None

        self.vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(self.vgg16.input, self.vgg16.get_layer('block3_conv2').output)

    def calculate_loss(self, generated_image, generated_img_features):
        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.loss += self.vgg_loss * tf_custom_l1_loss(self.ref_img_features, generated_img_features)

        # logcosh loss on image pixels
        if (self.pixel_loss is not None):
            self.loss += self.pixel_loss * tf_custom_logcosh_loss(self.ref_img, generated_image)

    def set_reference_images(self, images_list):
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(loaded_image)

        self.sess.run(tf.assign(self.ref_img, loaded_image))
        self.sess.run(tf.assign(self.ref_img_features, image_features))

    def optimize(self, vars_to_optimize, iterations=1000, learning_rate=0.01):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss