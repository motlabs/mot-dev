import numpy as np
import tensorflow as tf

from ..build import layers


def squeeze_layer(layer, num_outputs, name='squeeze_layer'):
    with tf.variable_scope(name):
        return layers.conv2d(layer, num_outputs, k_size=(1, 1), strides=(1, 1), name='squeeze')


def expand_layer(layer, num_outputs, name='expand_layer'):
    with tf.variable_scope(name):
        expand_1 = layers.conv2d(layer, num_outputs, k_size=(1, 1), strides=(1, 1), name='expand_1x1')
        expand_2 = layers.conv2d(layer, num_outputs, k_size=(3, 3), strides=(1, 1), name='expand_3x3')
        return tf.concat([expand_1, expand_2], -1)


def fire_module(layer, squeeze_depth, expand_depth, name='fire_module'):
    with tf.variable_scope(name):
        squeeze = squeeze_layer(layer, squeeze_depth)
        squeeze = layers.relu(squeeze)

        expand = expand_layer(squeeze, expand_depth)
        expand = layers.relu(expand)

        return expand


class BuildModel(object):
    def __init__(self, is_training, keep_prob, num_class):
        self.is_training = is_training
        self.keep_prob = keep_prob  # 0.5 in the paper
        self.num_class = num_class

    # Vanilla SqueezeNet
    def inference(self, layer):
        conv_1 = layers.conv2d(layer, 96, k_size=(3, 3), strides=(1, 1), name='conv_1')
        conv_1 = layers.relu(conv_1, name='act_1')

        fire_2 = fire_module(conv_1, 16, 64, name='fire_2')
        fire_3 = fire_module(fire_2, 16, 64, name='fire_3')
        fire_4 = fire_module(fire_3, 32, 128, name='fire_4')
        maxpool_1 = layers.max_pool(fire_4, k_size=(3, 3), strides=(2, 2), name='max_pool_1')

        fire_5 = fire_module(maxpool_1, 32, 128, name='fire_5')
        fire_6 = fire_module(fire_5, 32, 128, name='fire_6')
        fire_7 = fire_module(fire_6, 64, 256, name='fire_7')
        fire_8 = fire_module(fire_7, 64, 256, name='fire_8')
        maxpool_2 = layers.max_pool(fire_8, k_size=(3, 3), strides=(2, 2), name='max_pool_2')

        fire_9 = fire_module(maxpool_2, self.num_class, name='fire_9')
        dropout = layers.drop_out(fire_9, keep_prob=self.keep_prob, is_training=self.is_training, name='drop_out')
        conv_10 = layers.conv2d(dropout, self.num_class, k_size=(1, 1), strides=(1, 1), name='conv_10')
        logit = layers.glob_avg_pool(conv_10, name='max_pool_3')
        return logit

    # SqueezeNet with Simple bypass
    def inference_simple_bypass(self, layer):
        conv_1 = layers.conv2d(layer, 96, k_size=(3, 3), strides=(1, 1), name='conv_1')
        conv_1 = layers.relu(conv_1, name='act_1')

        fire_2 = fire_module(conv_1, 16, 64, name='fire_2')
        fire_3 = fire_module(fire_2, 16, 64, name='fire_3')
        conn_1 = tf.add(fire_2, fire_3, name='connection_1')
        fire_4 = fire_module(conn_1, 32, 128, name='fire_4')
        maxpool_1 = layers.max_pool(fire_4, k_size=(3, 3), strides=(2, 2), name='max_pool_1')

        fire_5 = fire_module(maxpool_1, 32, 128, name='fire_5')
        conn_2 = tf.add(maxpool_1, fire_5, name='connection_2')
        fire_6 = fire_module(conn_2, 32, 128, name='fire_6')
        fire_7 = fire_module(fire_6, 64, 256, name='fire_7')
        conn_3 = tf.add(fire_6, fire_7, name='connection_3')
        fire_8 = fire_module(conn_3, 64, 256, name='fire_8')
        maxpool_2 = layers.max_pool(fire_8, k_size=(3, 3), strides=(2, 2), name='max_pool_2')

        fire_9 = fire_module(maxpool_2, self.num_class, name='fire_9')
        conn_4 = tf.add(maxpool_2, fire_9, name='connection_4')
        dropout = layers.drop_out(conn_4, keep_prob=self.keep_prob, is_training=self.is_training, name='drop_out')
        conv_10 = layers.conv2d(dropout, self.num_class, k_size=(1, 1), strides=(1, 1), name='conv_10')
        logit = layers.glob_avg_pool(conv_10, name='max_pool_3')
        return logit
