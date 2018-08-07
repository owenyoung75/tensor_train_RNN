from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque


def rnn_with_feed_prev(cell,
                       inputs,
                       is_training,
                       config,
                       initial_state=None
                       ):
    cell_output = None
    outputs = []

    with tf.variable_scope("rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        input_size = int(inputs_shape[2])   # dim = 1 in examples
        inp_steps = config.inp_steps
        output_size = cell.output_size      # basic tf.RNNcell property
        acv_func = None
        
        if initial_state is None:
            # Note: cell already has been assigned with hidden-size = config.hidden_size
            initial_state = cell.zero_state(batch_size, dtype= tf.float32)
        state = initial_state

        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]
                    
            if not is_training and cell_output is not None and time_step >= inp_steps:
                # training finished, start forecasting using self-calculated sequence other than real data
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = output

            # Calculate use Tanh() function
            # state: (c,) h
            # cell_output: h
            (cell_output, state) = cell(inp, state)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output,
                                         input_size,
                                         activation_fn=acv_func)
                outputs.append(output)

    outputs = tf.stack(outputs, 1)
    return outputs, state



def _shift (input_list, new_item):
    """Update lag number of states"""
    output_list = copy.copy(input_list)
    output_list = deque(output_list)
    output_list.append(new_item)
    output_list.popleft()
    return output_list


def _list_to_states(states_list):
    # from [lag, layer]-index to [layer, lag]-index  --> for __call__ method
    num_layers = len(states_list[0])
    output_states = ()  # a tuple
    for layer in range(num_layers):
        output_state = ()
        for states in states_list:
            # c,h = states[layer] for LSTM
            # h   = states[layer] for RNN
            output_state += (states[layer],)
        output_states += (output_state,)
        # new cell has s*num_lags states
    return output_states


def tensor_rnn_with_feed_prev(cell,
                              inputs,
                              is_training,
                              config,
                              initial_states=None
                              ):
    outputs = []
    cell_output = None
    is_sample = is_training and initial_states is not None
    

    with tf.variable_scope("trnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        input_size = int(inputs_shape[2])
        output_size = cell.output_size
        inp_steps =  config.inp_steps
        acv_func = tf.sigmoid
        
        dist = Bernoulli(probs=config.sample_prob)
        samples = dist.sample(sample_shape=num_steps)
        
        if initial_states is None:
            initial_states =[]
            for lag in range(config.num_lags):
                initial_state =  cell.zero_state(batch_size, dtype=tf.float32)
                initial_states.append(initial_state)
        states_list = initial_states #list of high order states

#        for time_step in range(num_steps):
        for time_step in range(1):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]
            
            if is_sample and time_step > 0:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = tf.cond(tf.cast(samples[time_step], tf.bool),
                                  lambda: tf.identity(inp),
                                  lambda: output)
                    
            if not is_training and cell_output is not None and time_step >= inp_steps:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = output

            states = _list_to_states(states_list)
            (cell_output, state)=cell(inp, states)
            states_list = _shift(states_list, state)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output,
                                         input_size,
                                         activation_fn=acv_func)
                outputs.append(output)
    
    outputs = tf.stack(outputs, 1)
    return outputs, states_list



