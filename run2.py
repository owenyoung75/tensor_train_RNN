
"""Functions for downloading and reading time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base

from reader import read_data_sets
from train_config2 import *
from model_seq2seq import *
from trnn import *
from MERArnn import *




flags = tf.flags

flags.DEFINE_string('f', '', 'kernel2')
flags.DEFINE_string("model", "SymmetricTTRNN", "Model used for learning.")
flags.DEFINE_string("data_path", "./data.npy", "Data input directory.")
flags.DEFINE_integer("inp_steps", 3, "burn in steps")
flags.DEFINE_integer("out_steps", 5, "test steps")
flags.DEFINE_integer("hidden_size", 1, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay_rate", 0.8, "decay rate")
flags.DEFINE_integer("virtual_dim", 2, "dimension of virtual legs")
flags.DEFINE_integer("num_orders", 2, "order of polynomials")
flags.DEFINE_integer("num_lags", 2, "time-lag length")
flags.DEFINE_integer("num_layers", 2, "time-lag length")
flags.DEFINE_integer("batch_size", 10, "batch size")

FLAGS = flags.FLAGS
print('Flags configuration loaded ...')






# Training Parameters
config = TrainConfig()
config.hidden_size = FLAGS.hidden_size
config.learning_rate = FLAGS.learning_rate
config.decay_rate = FLAGS.decay_rate
config.virtual_dim = FLAGS.virtual_dim
config.num_orders = FLAGS.num_orders
config.num_lags = FLAGS.num_lags
config.num_layers = FLAGS.num_layers
config.inp_steps = FLAGS.inp_steps
config.out_steps = FLAGS.out_steps
config.batch_size = FLAGS.batch_size


# Training Parameters
training_steps = config.training_steps
batch_size = config.batch_size
display_step = 500
inp_steps = config.inp_steps
out_steps = config.out_steps


# Read Dataset
dataset, stats = read_data_sets(FLAGS.data_path, True, inp_steps, out_steps)
num_input = stats['num_input']  # dataset data input (time series dimension: 1)
num_steps = stats['num_steps']


# Print training config
print('-'*120)
print('|input steps|', inp_steps,
      '|out steps|', out_steps,
      '|hidden size|', config.hidden_size,
      '|learning rate|', config.learning_rate,
      '|orders|', config.num_orders,
      '|virtual-D|', config.virtual_dim,
      '|time lag|', config.num_lags,
      '|layer number|', config.num_layers
      )
print('-'*120)



batch_x, batch_y, batch_z = dataset.train.next_batch(batch_size)
X = tf.convert_to_tensor(batch_x, dtype=tf.float32)
Y = tf.convert_to_tensor(batch_y, dtype=tf.float32)

X = X[:, 0:5, :]
Y = Y[:, 0:5, :]
Z = Y

Model = globals()[FLAGS.model]


with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
    test_pred = Model(X, Y, False,  config)
#    print(test_pred)
