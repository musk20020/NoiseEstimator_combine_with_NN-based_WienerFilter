import tensorflow as tf
import h5py
import numpy as np
import scipy
from multiprocessing import Pool
from utils import _gen_VAD_training_data_runtime
from glob import iglob
from functools import partial
import csv
import time
import os
from os.path import join
from tqdm import tqdm
import xlrd, xlwt

# from gru_cus_cell import EGGRUCell
# from gru_poly_cell import GRUPolyCell

tqdm.monitor_interval = 0
# from utils_2 import np_REG_batch, search_wav, wav2spec, spec2wav, copy_file, np_batch, get_embedding, get_dist_table
from utils import np_REG_batch, search_wav, copy_file, np_batch, get_embedding, get_dist_table, _gen_audio
from sklearn.utils import shuffle
import tensorflow_utils as tfu
import wandb


# eps = np.finfo(np.float32).epsilon()

class VAD:

    def __init__(self, log_path, saver_path, date, gpu_num, note, config):

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
        self.log_path = log_path
        self.saver_path = saver_path
        self.saver_dir = '{}_{}/{}'.format(self.saver_path, note, date)
        self.eps = np.finfo(np.float32).eps
        self.saver_name = join(
            self.saver_dir, 'best_saver_{}'.format(note))
        self.tb_dir = '{}_{}/{}'.format(self.log_path, note, date)
        self.config = config

        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir)

    def build(self, reuse):
        config = {
            "batch_size": self.config.batch_size,
            "filter_h": 5,
            "filter_w": 5,
            "mel_freq_num": self.config.mel_freq_num,
            "l1_output_num": 40,
            "l2_output_num": 20,
            "l3_output_num": 10,
        }

        self.name = 'VAD'
        input_dimension = 24  # RNN input
        output_dimension = 1

        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            with tf.variable_scope('Intputs'):
                self.input = tf.placeholder(
                    tf.float32, shape=[None, input_dimension, self.config.stoi_correlation_time, 1], name='x_norm')

            with tf.variable_scope('featureExtractor', reuse=tf.AUTO_REUSE):
                layer_1 = tfu._add_conv_layer(self.input, layer_num='1', filter_h=config["filter_h"],
                                              filter_w=config["filter_w"], input_c=1,
                                              output_c=config["l1_output_num"], dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME',
                                              trainable=True)  # [N, 126, t-2, 512]
                layer_2 = tfu._add_conv_layer(layer_1, layer_num='2', filter_h=config["filter_h"],
                                              filter_w=config["filter_w"],
                                              input_c=config["l1_output_num"],
                                              output_c=config["l2_output_num"], dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME',
                                              trainable=True)  # [N, 62, t-4, 512]
                layer_3 = tfu._add_conv_layer(layer_2, layer_num='3', filter_h=config["filter_h"],
                                              filter_w=1, input_c=config["l2_output_num"],
                                              output_c=config["l3_output_num"], dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME',
                                              trainable=True)  # [N, 124, t-4, 128]
                reshape = tf.reshape(tf.transpose(layer_3, perm=[0, 2, 3, 1]),
                                     [-1, self.config.stoi_correlation_time,
                                      config["l3_output_num"] * input_dimension])
                self.output = tfu._add_3dfc_layer(reshape, config["l3_output_num"] * input_dimension, 1,
                                             '4', activate_function=tf.nn.sigmoid, trainable=True, keep_prob=1)
            self.saver = tf.train.Saver()

    def init(self):
        sess = tf.Session()
        model_path = "/AudioProject/nb_denoise/model/220126/"
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is not None:
            self.saver_pre.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("model not found")
        return sess

    def predict(self, sess, mag):
        output = sess.run(self.output, feed_dict={self.input:mag})
        return output


