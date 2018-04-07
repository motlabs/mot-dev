import tensorflow as tf

from tensorflow.contrib.layers import batch_norm as batch_normalization


initializer = tf.contrib.layers.xavier_initializer()


def pRelu(x, name="P_relu"):
    with tf.name_scope(name):
        alpha = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)

act_fn = pRelu

def conv2d(layer_input, output_dim, k_size=(3, 3), strides=(1, 1), activation_fn=act_fn, normalization=True, is_training=True, name='conv2d'):
    with tf.variable_scope(name):
        weight = tf.get_variable('w', list(k_size) + [layer_input.get_shape()[-1], output_dim], initializer=initializer)
        conv = tf.nn.conv2d(layer_input, weight, strides=[1] + list(strides) + [1], padding='SAME', name='conv')
        bias = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
        conv = tf.nn.bias_add(conv, bias)

        if normalization is not None:
            conv = batch_norm(conv, train_phase=is_training, name='batch_norm')
        if activation_fn is not None:
            conv = activation_fn(conv, 'act')

        return conv


def max_pool(input_, k_size=(2, 2), strides=(2, 2), padding='SAME', name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(input_, ksize=[1] + list(k_size) + [1], strides=[1] + list(strides) + [1], padding=padding)


def drop_out(x, keep_prob, is_training, name="drop_out"):
    with tf.variable_scope(name):
        dropout = tf.nn.dropout(x, keep_prob)
        full_dropout = tf.nn.dropout(x, 1.)
        return tf.cond(is_training, lambda: dropout, lambda: full_dropout)


def batch_norm(inputs, train_phase=True, name="batch_norm"):
    if type(train_phase) == bool:
        train_phase = tf.constant(train_phase, dtype=tf.bool, shape=[])

    train_bn = batch_normalization(inputs=inputs, decay=0.9, updates_collections=None, zero_debias_moving_mean=True,
                                   is_training=True, reuse=None, scale=True, epsilon=1e-5, trainable=True, scope=name)
    test_bn = batch_normalization(inputs=inputs, decay=0.9, updates_collections=None, zero_debias_moving_mean=True,
                                  is_training=False, reuse=True, scale=True, epsilon=1e-5, trainable=True, scope=name)

    return tf.cond(train_phase, lambda: train_bn, lambda: test_bn)


class BuildModel(object):
    def __init__(self, is_training, keep_prob, num_class):
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.num_class = num_class

    def inference(self, layer):
        layer = conv2d(layer, 64, is_training=self.is_training, name='conv1')
        layer = conv2d(layer, 64, is_training=self.is_training, name='conv2')
        layer = max_pool(layer, name='max_pool1')

        layer = conv2d(layer, 128, is_training=self.is_training, name='conv3')
        layer = conv2d(layer, 128, is_training=self.is_training, name='conv4')
        layer = max_pool(layer, name='max_pool2')

        layer = conv2d(layer, 256, is_training=self.is_training, name='conv5')
        layer = conv2d(layer, 256, is_training=self.is_training, name='conv6')
        layer = conv2d(layer, 256, is_training=self.is_training, name='conv7')
        layer = max_pool(layer, name='max_pool3')

        layer = conv2d(layer, 512, is_training=self.is_training, name='conv8')
        layer = conv2d(layer, 512, is_training=self.is_training, name='conv9')
        layer = conv2d(layer, 512, is_training=self.is_training, name='conv10')
        layer = max_pool(layer, name='max_pool4')

        layer = conv2d(layer, 512, is_training=self.is_training, name='conv11')
        layer = conv2d(layer, 512, is_training=self.is_training, name='conv12')
        layer = conv2d(layer, 512, is_training=self.is_training, name='conv13')
        net = max_pool(layer, name='max_pool5')

        # Fully Connected
        net = conv2d(net, 4096, k_size=(7, 7), name='fc1')
        net = drop_out(net, keep_prob=self.keep_prob, is_training=self.is_training, name='dropout_1')
        net = conv2d(net, 4096, k_size=(1, 1), name='fc2')

        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

        net = drop_out(net, keep_prob=self.keep_prob, is_training=self.is_training, name='dropout_2')
        net = conv2d(net, self.num_class, (1, 1), activation_fn=None, normalization=None, name='conv14')
        net = tf.squeeze(net, [1, 2], name='output')
        return net
