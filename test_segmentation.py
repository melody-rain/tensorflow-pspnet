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

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]

num_class = 21
palette = palette[:num_class]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, num_class)



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

        #####################################
        # Select the preprocessing function #
        #####################################
        input_image_ori = scipy.misc.imread(FLAGS.image)
        H, W = input_image_ori.shape[0], input_image_ori.shape[1]
        input_image = scipy.misc.imresize(input_image_ori, (473, 473))

        image_X = tf.placeholder(tf.uint8, input_image.shape)
        # image = image_X - [_R_MEAN, _G_MEAN, _B_MEAN]
        image = tf.image.convert_image_dtype(image_X, dtype=tf.float32)
        images = tf.expand_dims(image, axis=[0])

        ####################
        # Define the model #
        ####################
        logits, _, _ = network_fn(images)
        logits = slim.softmax(logits)

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

        import time
        before = time.time()
        input_img, logit = sess.run([image, logits], feed_dict={image_X: input_image})
        print(time.time() - before)

        before = time.time()
        input_img, logit = sess.run([image, logits], feed_dict={image_X: input_image})
        print(time.time() - before)

        logit = logit[0]

        p = np.argmax(logit, axis=2)
        p = p.astype(np.uint8)

        p = scipy.misc.imresize(p, (H, W))

        fig = plt.figure()
        ax = fig.add_subplot('121')
        ax.imshow(input_image_ori)
        ax = fig.add_subplot('122')
        ax.matshow(p, vmin=0, vmax=21, cmap=my_cmap)
        plt.show()

        m = p == 1
        m = m.astype(np.uint8)
        masked = input_image_ori * m[:, :, np.newaxis]

        fig = plt.figure()
        ax = fig.add_subplot('121')
        ax.imshow(input_image_ori)
        ax = fig.add_subplot('122')
        ax.imshow(masked)  # , cmap='gray')
        plt.show()


if __name__ == '__main__':
    tf.app.run()
