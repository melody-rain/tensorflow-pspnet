#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import cv2
from matplotlib import colors as mpl_colors
import json
from PIL import PngImagePlugin, Image

from datasets import dataset_factory
from nets import nets_factory
import os
import cv2
from os.path import exists, join, split, splitext

slim = tf.contrib.slim

__author__ = 'Soonmin Hwang'
__email__ = 'smhwang@rcv.kaist.ac.kr'
__description__ = 'This code is a modified version of F.Yus implementation. \
                    (https://github.com/fyu/dilated.git) '

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

tf.app.flags.DEFINE_string(
    'image_list', '',
    'image list')

tf.app.flags.DEFINE_string(
    'data_root', '',
    'data root')

FLAGS = tf.app.flags.FLAGS

num_class = 150

color_info_file = '/home/melody/develop/caffe-segmentation/misc/palette/ade20k.json'
with open(color_info_file) as fd:
    data = json.load(fd)
    palette = np.array([(0, 0, 0)] + data['palette'][:num_class], dtype=np.uint8)

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

IMG_MEAN = np.array((_B_MEAN, _G_MEAN, _R_MEAN), dtype=np.float32)

result_dir = 'results'

ignore_label = 0


def preprocess2(img, h, w):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    pad_img = tf.expand_dims(img, dim=0)
    pad_img = tf.image.resize_bilinear(pad_img, (h, w), align_corners=True)

    return pad_img


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = tf.train.get_or_create_global_step()

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=num_class,
            is_training=False)

        image_paths = [line.strip() for line in open(FLAGS.image_list, 'r')]
        image_sets = [tuple(a_line.strip().split()) for a_line in image_paths]

        image_names = [os.path.join(FLAGS.data_root, p[0]) for p in image_sets]
        image_labels = [os.path.join(FLAGS.data_root, p[1]) for p in image_sets]

        predictions_all = np.zeros((len(image_names), 360, 480), dtype=int)
        gts_all = np.zeros((len(image_names), 360, 480), dtype=int)
        current_ms = lambda: int(round(time.time() * 1000))

        # -----------------------------------------------------<
        image_filename = tf.placeholder(dtype=tf.string)
        anno_filename = tf.placeholder(dtype=tf.string)

        img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
        anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)

        img.set_shape([None, None, 3])
        anno.set_shape([None, None, 1])

        img_shape = tf.shape(img)
        h, w = 473, 473  # (tf.maximum(crop_size[0], shape[0]), tf.maximum(crop_size[1], shape[1]))
        images = preprocess2(img, h, w)

        ####################
        # Define the model #
        ####################
        net, end_points = network_fn(images)
        raw_output_up = net

        raw_output_up = tf.image.resize_bilinear(raw_output_up, size=[img_shape[0], img_shape[1]], align_corners=True)
        raw_output_up = tf.argmax(raw_output_up, dimension=3)

        pred_flatten = tf.reshape(raw_output_up, [-1, ])
        raw_gt = tf.reshape(anno, [-1, ])
        indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        pred = tf.gather(pred_flatten, indices)

        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_class + 1)

        accuracy, update_op_a = tf.contrib.metrics.streaming_accuracy(pred, gt)

        if FLAGS.moving_average_decay > 0:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        sess = tf.Session()

        saver = tf.train.Saver(variables_to_restore)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        sess.run(init_op)
        saver.restore(sess, checkpoint_path)

        # ----------------------------------------------------->

        for i in xrange(len(image_names)):
            print('Predicting', image_names[i])

            cur_time = current_ms()
            raw_output, _, _ = sess.run([raw_output_up, update_op, update_op_a],
                                        feed_dict={image_filename: image_names[i], anno_filename: image_labels[i]})
            print(raw_output)
            cur_time = current_ms() - cur_time
            print('time: ', cur_time)

            prediction = raw_output[0]
            out_path = join(result_dir,
                            splitext(image_names[i].split('/')[-1])[0] + '.png')
            print('Writing', out_path)
            im = Image.fromarray(prediction.astype(np.uint8), mode='P')
            im.putpalette(palette.flatten())
            im.save(out_path)

        print('mIoU: {:04f}'.format(sess.run(mIoU)))
        print('Pixel_ACC: {:04f}'.format(sess.run(accuracy)))

    print('================================')
    print('All results are generated.')
    print('================================')


if __name__ == '__main__':
    tf.app.run()
