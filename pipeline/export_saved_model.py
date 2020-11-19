import tensorflow as tf
import os


model_path = '../../cache/finetuned_resnet.h5'
export_path = '../../models/ResNet/1/'

model = tf.keras.models.load_model(model_path)
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

print('\nSaved model successfully...')