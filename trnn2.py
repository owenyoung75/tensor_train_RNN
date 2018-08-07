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




class SymmetricMPS_RNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._num_orders = num_orders
        self._virtual_dim = virtual_dim
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
        new_h = Symmetric_MPS_wavefn(inputs,
                                     states,
                                     self.output_size,
                                     self._num_orders,
                                     self._virtual_dim,
                                     True)
        new_h = self._activation(new_h)
        return  new_h, new_h




class SymmetricMPS_LSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._num_orders = num_orders
        self._virtual_dim = virtual_dim
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
        concat = Symmetric_MPS_wavefn(inputs,
                                      hs,
                                      meta_variable_size,
                                      self._num_orders,
                                      self._virtual_dim,
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


def _periodic_MPS_contraction(states_tensor, MPS_tensors):
    
    virtual  = "abcdefghijklm"
    physical = "nopqrstuvwxy"    
    num_orders = states_tensor.shape.ndims - 1

    def _get_einsum(i, old_legs):
        A_legs = virtual[i] + physical[i] + virtual[i+1]
        if i == 0:
            new_legs = old_legs + virtual[i] + virtual[i+1]
        elif i == num_orders:
            A_legs = A_legs.replace(A_legs[2], virtual[0])
            new_legs = "z" + "0" + physical[i]
        else:
            new_legs = old_legs.replace(old_legs[-1], virtual[i+1])
        new_legs = new_legs.replace(new_legs[1], "")
        ein = A_legs + "," + old_legs + "->" + new_legs
        return ein, new_legs

    out_h = states_tensor
    legs = "z" + physical[:num_orders]

    for i in range(num_orders):
        einsum, legs = _get_einsum(i, legs)
        out_h = tf.einsum(einsum, MPS_tensors[0], out_h)
    
    einsum, legs = _get_einsum(num_orders, legs)
    out_h = tf.einsum(einsum, MPS_tensors[1], out_h)
    return out_h



def periodic_MPS_contraction(states_tensor, MPS_tensors):
    num_orders = states_tensor.shape.ndims - 1

    out_h = tf.tensordot(states_tensor, MPS_tensors[0], [[num_orders], [1]])
    for _ in range(num_orders-1):
        out_h = tf.tensordot(out_h, MPS_tensors[0], [[-3, -1],[1, 0]])

    out_h = tf.tensordot(out_h, MPS_tensors[1], [[1, 2],[2, 0]])
    return out_h





def Symmetric_MPS_wavefn(inputs,
                         states,
                         output_size,
                         num_orders,
                         virtual_dim,
                         bias,
                         bias_start=0.0
                         ):
    num_lags = len(states)
    state_size = states[0].get_shape()[1].value         # hidden_size, i.e. h_{t} dimension
    total_state_size = (state_size * num_lags + 1 )     # [HL + 1]
    
    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables

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
    
    print(states_tensor.shape)
    large_Tn = []
    for order in range(num_orders):
        large_Tn.append(states_vector)
    large_Tn = tf.transpose(tf.convert_to_tensor(large_Tn), perm=[1,0,2])
    print(large_Tn.shape)
    filter_ = tf.get_variable("conv_filter",shape=[2, 3, 1], trainable=False)
    convolved = tf.nn.conv1d(large_Tn, filters=filter_, stride=2, padding="VALID")
    print(convolved.shape)

    
    """ construct a big serialized variable for all MPS parameters """
    tsA_size   = total_state_size * virtual_dim * virtual_dim
    tsOut_size = output_size * virtual_dim * virtual_dim
    tsr = vs.get_variable("weights_h",
                          tsA_size + tsOut_size,
                          trainable = True) # h_z x h_z... x output_size
    
    ts_A = tf.slice(tsr, [0], [tsA_size] )
    ts_A = tf.reshape(ts_A, [virtual_dim,
                             total_state_size,
                             virtual_dim])
    ts_Out = tf.slice(tsr, [tsA_size], [tsOut_size] )
    ts_Out = tf.reshape(ts_Out, [virtual_dim,
                                 output_size,
                                 virtual_dim])
                                 
    MPS = [ts_A, ts_Out]

    weights_x = vs.get_variable("weights_x",
                                [input_size, output_size],
                                trainable = True)

    out_h = periodic_MPS_contraction(states_tensor, MPS)
    
    out_x = tf.matmul(inputs, weights_x)
    
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])
    return nn_ops.bias_add(res,biases)


