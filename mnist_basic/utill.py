import tensorflow as tf
from tensorflow.python.client import device_lib
import sys
import os
import configparser
import ast


ckpt_dir = 'save_point/'
model_name = 'model'


def progress_bar(total, progress, state_msg):
    """
    Displays or updates a console progress bar.
    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"

    block = int(round(barLength * progress))
    progress_bar = "\r [{}] {:.0f}% -> {}{}".format("#" * block + "-" * (barLength - block),
                                                    round(progress * 100, 0),
                                                    state_msg,
                                                    status)
    sys.stdout.write(progress_bar)
    sys.stdout.flush()


def save_ckpt(sess, step):
    checkpoint_dir = os.path.join(os.getcwd(), ckpt_dir)
    if not os.path.exists(checkpoint_dir):
        # Checkpoint save path
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(max_to_keep=3)
    save_name = saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    tf.train.write_graph(sess.graph_def, checkpoint_dir, 'graph.pbtxt', as_text=True)
    print("Check Point Saved : ", save_name)