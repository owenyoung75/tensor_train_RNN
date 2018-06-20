from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque




class MPS_RNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 rank_vals,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._rank_vals = rank_vals
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
    
    def __call__(self, inputs, states):
        """this method is inheritated, and always calculate layer by layer"""
        output = MPS_einsum(inputs,
                            states,
                            self.output_size,
                            self._rank_vals,
                            True)
        new_h = self._activation(output)
        return  new_h, new_h




class MPS_LSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 rank_vals,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._rank_vals = rank_vals
        self._forget_bias = forget_bias
        self._state_is_tuple= state_is_tuple
        self._activation = activation
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)
    
    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states):
        """this method is inheritated, and always calculate layer by layer"""
        """Now we have multiple states, state->states"""
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        # states: size = time_lag
        if self._state_is_tuple:
            hs = ()
            for state in states:
                c, h = state    # c and h: tensor_size = (batch_size, hidden_size)
                hs += (h,)      # hs : size = time_lag, i.e. time_lag * (batch_size, hidden_size)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state,
                                       num_or_size_splits=2,
                                       axis=1)
                hs += (h,)
        
        meta_variable_size = 4 * self.output_size
        concat = MPS_einsum(inputs,
                            hs,
                            meta_variable_size,
                            self._rank_vals,
                            True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat,
                                     num_or_size_splits=4,
                                     axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state




def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]

def _outer_product(batch_size, tensor, vector):
    """tensor-vector outer-product"""
    tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
    vector_flat = tf.expand_dims(vector, 1)
#    print(tensor.shape, '||', vector_flat.shape)
    res = tf.matmul(tensor_flat, vector_flat)
    new_shape =  [batch_size]+_shape_value(tensor)[1:]+_shape_value(vector)[1:]
    res = tf.reshape(res, new_shape )
    return res


def MPS_tensor_contraction(states_tensor, MPS_tensors):
    virtual  = "abcdefghijklm"
    physical = "nopqrstuvwxy"

    def _get_einsum(i, old_legs):
        A_legs = virtual[i] + physical[i] + virtual[i+1]
        if i==0:
            new_legs = old_legs + virtual[i] + virtual[i+1]
        else:
            new_legs = old_legs.replace(old_legs[-1], virtual[i+1])
        new_legs = new_legs.replace(new_legs[1], "")
        ein = A_legs + "," + old_legs + "->" + new_legs
        return ein, new_legs

    num_orders = len(MPS_tensors)
    out_h = states_tensor
    legs = "z" + physical[:num_orders]

    for i in range(num_orders):
        einsum, legs = _get_einsum(i, legs)
#        print(einsum)
        out_h = tf.einsum(einsum, MPS_tensors[i], out_h)
    out_h = tf.squeeze(out_h, [1])
    return out_h


def MPS_einsum(inputs,
               states,
               output_size,
               rank_vals,
               bias,
               bias_start=0.0
               ):
    num_orders = len(rank_vals)+1 # alpha_1 to alpha_{K-1}, control the number of copies
    num_lags = len(states)
    batch_size = tf.shape(inputs)[0] 
    state_size = states[0].get_shape()[1].value # hidden_size, i.e. h_{t} dimension
    input_size= inputs.get_shape()[1].value     # dimension of variables
    total_state_size = (state_size * num_lags + 1 )     # [HL + 1]

    """construct augmented state tensor"""
    states_vector0 = tf.concat(states, 1)    # serialize all h at different time_lags
    states_vector = tf.concat([states_vector0, tf.ones([batch_size, 1])], 1) # add the 0th-order: 1
    # make higher order tensors S_{t-1} for a layer
    # vector : tensor_size = ( batchs, HL+1 )
    states_tensor = states_vector
    for order in range(num_orders-1):
        states_tensor = _outer_product(batch_size,
                                       states_tensor,
                                       states_vector)




    # physical_dim: [(HL+1), (HL+1), ... , (HL+1)]
    mat_dims = np.ones((num_orders,)) * total_state_size
    # virtual_dim * output_dim
    mat_ranks = np.concatenate(([1],
                                rank_vals,
                                [output_size]
                                ))

    # Each factor A is a 3-tensor, with dimensions:
    # [mat_rank[i-1], hidden_size, mat_rank[i] ]
    # total-dim of A is ( mat_rank[i-1] * hidden_size * mat_rank[i] )
    mat_ps = np.cumsum(np.concatenate(([0],
                                       mat_ranks[:-1] * mat_dims * mat_ranks[1:]
                                       )),
                       dtype=np.int32)
    # totoal number of approximating parameters
    mat_size = mat_ps[-1]  # = (P-1) * (HL+1) * R^2
                           #     + 1 * (HL+1) * R*out_hidden_size
    """ construct a big serialized variable for all MPS parameters """
    mat = vs.get_variable("weights_h",
                          mat_size,
                          trainable = True) # h_z x h_z... x output_size
    MPS = []
    for i in range(num_orders):
        # Fetch the weights of factor A^i from serialized variable weights_h.
        mat_A = tf.slice(mat,
                         [mat_ps[i]],
                         [mat_ps[i + 1] - mat_ps[i]])
        mat_A = tf.reshape(mat_A, [mat_ranks[i],
                                   total_state_size,
                                   mat_ranks[i + 1]])
        MPS.append(mat_A)

    weights_x = vs.get_variable("weights_x",
                                [input_size, output_size],
                                trainable = True)

    out_h = MPS_tensor_contraction(states_tensor, MPS)
    out_x = tf.matmul(inputs, weights_x)
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])

    return nn_ops.bias_add(res,biases)


