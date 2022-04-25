import tensorflow as tf
import h5py
import numpy as np
import scipy
from multiprocessing import Pool
from utils import _gen_nnFilter_training_data_runtime
from glob import iglob
from functools import partial
import csv
import time
import os
from os.path import join
from tqdm import tqdm
import xlrd, xlwt

tqdm.monitor_interval = 0
from utils import np_REG_batch, search_wav, copy_file, np_batch, get_embedding, get_dist_table, _gen_audio
from sklearn.utils import shuffle
import tensorflow_utils as tfu
import audio_processing as ap
import librosa


# eps = np.finfo(np.float32).epsilon()

class REG:

    def __init__(self, gpu_num):

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    def build(self, reuse):
        config = {
            "batch_size": 1,
            "filter_h": 7,
            "filter_w": 3,
            "mel_freq_num": 24,
            "l1_output_num": 2,
            "l2_output_num": 129,
            "l3_output_num": 129,
        }

        self.name = 'NEWF'
        input_dimension = 129  # RNN input

        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            with tf.variable_scope('Intputs'):
                self.x_noisy_norm = tf.placeholder(
                    tf.float32, shape=[1, input_dimension, None, 1], name='x_norm')

            with tf.variable_scope('featureExtractor', reuse=tf.AUTO_REUSE):
                layer_1 = tfu._add_conv_layer(self.x_noisy_norm, layer_num='1', filter_h=config["filter_h"],
                                              filter_w=config["filter_w"], input_c=1,
                                              output_c=config["l1_output_num"], dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME',
                                              trainable=True)
                reshape = tf.reshape(tf.transpose(layer_1, perm=[0, 2, 3, 1]),
                                     [1, -1, config["l1_output_num"] * input_dimension])
                layer_2 = tfu._add_3dfc_layer(reshape, config["l1_output_num"] * 129, config["l2_output_num"],
                                              '2', activate_function=tf.nn.tanh, trainable=True, keep_prob=1)
                self.mask = tfu._add_3dfc_layer(layer_2, config["l2_output_num"], config["l3_output_num"],
                                           '3', activate_function=tf.nn.sigmoid, trainable=True, keep_prob=1)

            var_list = tf.all_variables()
            self.saver_NEWF = tf.train.Saver(var_list=[v for v in var_list if 'NEWF' in v.name])

    def init(self, model_path):
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is not None:
            self.saver_NEWF.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("NEWF_model not found")
        return sess

    def inference(self, sess, mag):
        # y, _ = librosa.load(audio_file, 16000)
        # y = ap.second_order_filter_freq(y)
        mask = sess.run(self.mask, feed_dict={self.x_noisy_norm:mag})
        return mask