from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import scipy
import scipy.misc as misc
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

from datasets import dataset_factory
from nets import nets_factory
import os
import cv2
from PIL import Image
import json
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'model_name', 'pspnet_v1_101', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'image', None, 'Test image')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
FLAGS = tf.app.flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
num_class = 21

color_info_file = 'pascal_voc.json'
with open(color_info_file) as fd:
    data = json.load(fd)
    palette = np.array(data['palette'][:num_class], dtype=np.uint8)

mean_pixel = np.array([104.008, 116.669, 122.675], dtype=np.float)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = 0#tf.train.get_or_create_global_step()

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=num_class,
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        file_name = os.path.basename(FLAGS.image)
        file_name = file_name[:file_name.rfind('.')]
        # preprocess images
        image_ori = cv2.imread(FLAGS.image).astype(np.float32) - mean_pixel
        img_shape = image_ori.shape

        img = cv2.resize(image_ori, (473, 473), interpolation=cv2.INTER_CUBIC)

        image_h = tf.placeholder(tf.float32, img.shape)
        images = tf.expand_dims(image_h, axis=0)

        ####################
        # Define the model #
        ####################
        net, end_points = network_fn(images)

        raw_output_up = net

        raw_output_up = tf.image.resize_bilinear(raw_output_up, size=[img_shape[0], img_shape[1]], align_corners=True)
        raw_output_up = tf.argmax(raw_output_up, dimension=3)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        sess.run(init)

        restore_var = tf.global_variables()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        saver = tf.train.Saver(var_list=restore_var)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))

        in_img, preds = sess.run([images, raw_output_up], feed_dict={image_h: img})
        preds = preds[0]

        im = Image.fromarray(preds.astype(np.uint8), mode='P')
        im.putpalette(palette.flatten())
        im.save('pred_{}.png'.format(file_name))


if __name__ == '__main__':
    tf.app.run()
