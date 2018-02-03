from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import pspnet_v1

slim = tf.contrib.slim

networks_map = {'pspnet_v1_50': pspnet_v1.pspnet_v1_50,
                'pspnet_v1_101': pspnet_v1.pspnet_v1_101}

arg_scopes_map = {'pspnet_v1_50': pspnet_v1.pspnet_arg_scope,
                  'pspnet_v1_101': pspnet_v1.pspnet_arg_scope}


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)

    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
