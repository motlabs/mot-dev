import tensorflow as tf

from tensorflow.contrib.layers import batch_norm as batch_normalization


initializer = tf.contrib.layers.xavier_initializer()
act_fn = pRelu


def relu(layer, name='relu'):
    with tf.name_scope(name):
        return tf.nn.relu(layer)


def pRelu(x, name="P_relu"):
    with tf.name_scope(name):
        alpha = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)


def conv2d(layer_input, output_dim, k_size=(3, 3), strides=(1, 1), activation_fn=None, normalization=None, is_training=True, name='conv2d'):
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


def glob_avg_pool(layer, padding='SAME', name='glob_avg_pool'):
    with tf.name_scope(name):
        input_shape = layer.get_shape()
        return tf.nn.avg_pool(layer, ksize=[1, input_shape[1], input_shape[2], 1], strides=[1, 1, 1, 1], padding=padding)


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