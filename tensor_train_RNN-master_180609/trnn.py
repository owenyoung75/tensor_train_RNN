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
#from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
#from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.contrib.rnn import LSTMStateTuple

import numpy as np
import copy
from collections import deque


class TensorLSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self, config, forget_bias=1.0, state_is_tuple=True, activation=tanh, reuse=None):
#        super(TensorLSTMCell, self).__init__(_reuse=reuse)
#        super().__init__(_reuse=reuse)
        super(TensorLSTMCell, self).__init__()
        self._hidden_size = config.hidden_size
        self._permute_series_list = config.permute_series_list
        self._rank_vals = config.rank_vals
        self._expanded_hidden_size = config.expanded_hidden_size
        self._poly_order = config.poly_order
#        self._constant_permute_tensors = config.constant_permute_tensors

        self._forget_bias = forget_bias
        self._state_is_tuple= state_is_tuple
        self._activation = activation
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._hidden_size, self._hidden_size)
                if self._state_is_tuple else 2 * self._hidden_size)
    
    @property
    def output_size(self):
        return self._hidden_size

    def __call__(self, inputs, states):
        """Now we have multiple states, state->states"""
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            hs = ()
            for state in states:
            # every state is a tuple of (c,h)
                c, h = state
                hs += (h,)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
                hs += (h,)

        #four copies of hidden_size, as i,j,f, and o.
        concat = vicissitude(inputs, hs, self._permute_series_list, self._hidden_size, self._expanded_hidden_size, self._poly_order, self._rank_vals, 4 * self._hidden_size, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state



    

def _linear(args, output_size, bias, bias_start=0.0):
    total_arg_size = 0
    shapes= [a.get_shape() for a in args]
    for shape in shapes:
        total_arg_size += shape[1].value
    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable("weights", [total_arg_size, output_size], dtype=dtype)
        """y = [batch_size x total_arg_size] * [total_arg_size x output_size]"""
        res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            biases = vs.get_variable("biases", [output_size], dtype=dtype)
    return  nn_ops.bias_add(res,biases)

def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]

def _outer_product(batch_size, tensor, vector):
    """tensor-vector outer-product"""
    tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
    vector_flat = tf.expand_dims(vector, 1)
    res = tf.matmul(tensor_flat, vector_flat)
    new_shape =  [batch_size]+_shape_value(tensor)[1:]+_shape_value(vector)[1:]
    res = tf.reshape(res, new_shape )
    return res

def sparse_tensor_dense_tensordot(sp_a, b, axes, name=None):
    r"""Tensor contraction of a and b along specified axes.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
    The lists `a_axes` and `b_axes` specify those pairs of axes along which to
    contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
    as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
    `a_axes` and `b_axes` must have identical length and consist of unique
    integers that specify valid axes for each of the tensors.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.
    Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
    is equivalent to matrix multiplication.
    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.
    Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
    Args:
        a: `SparseTensor` of type `float32` or `float64`.
        b: `Tensor` with the same type as `a`.
        axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
         If axes is a scalar, sum over the last N axes of a and the first N axes
         of b in order.
         If axes is a list or `Tensor` the first and second row contain the set of
         unique integers specifying axes along which the contraction is computed,
         for `a` and `b`, respectively. The number of axes for `a` and `b` must
         be equal.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` with the same type as `a`.
    Raises:
        ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
        IndexError: If the values in axes exceed the rank of the corresponding
            tensor.
    """

    def _tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                    axes < 0, tf.int32) * (
                            axes + rank_a)
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                                list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                                range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, compat.integral_types) and \
                    isinstance(b_axes, compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                        "Different number of contraction axes 'a' and 'b', %s != %s." %
                        (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    def _sparse_tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
#            print("??")
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
#            print(a, perm)
            permed_a = tf.sparse_transpose(a, perm)
#            print("???")
            reshaped_a = tf.sparse_reshape(permed_a, new_shape)

            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                    axes < 0, tf.int32) * (
                            axes + rank_a)
            #print(sess.run(rank_a), sess.run(axes))
#            print("??")
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
#            print("???")
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.sparse_reshape(tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                                list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                                range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                        "Different number of contraction axes 'a' and 'b', %s != %s." %
                        (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    with tf.name_scope(name, "SparseTensorDenseTensordot", [sp_a, b, axes]) as name:
#         a = tf.convert_to_tensor(a, name="a")
#        print("more warning 1")
        b = tf.convert_to_tensor(b, name="b")
#        print("more warning 2")
        sp_a_axes, b_axes = _sparse_tensordot_axes(sp_a, axes)
#        print("more warning 3")
        sp_a_reshape, sp_a_free_dims, sp_a_free_dims_static = _sparse_tensordot_reshape(sp_a, sp_a_axes)
#        print("more warning 4")
        b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
                b, b_axes, True)
        
        ab_matmul = tf.sparse_tensor_dense_matmul(sp_a_reshape, b_reshape)
        if isinstance(sp_a_free_dims, list) and isinstance(b_free_dims, list):
            return tf.reshape(ab_matmul, sp_a_free_dims + b_free_dims, name=name)
        else:
            sp_a_free_dims = tf.convert_to_tensor(sp_a_free_dims, dtype=tf.int32)
            b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.int32)
            product = tf.reshape(
                    ab_matmul, tf.concat([sp_a_free_dims, b_free_dims], 0), name=name)
            if sp_a_free_dims_static is not None and b_free_dims_static is not None:
                product.set_shape(sp_a_free_dims_static + b_free_dims_static)
            return product

            
 
            
            
            
            
            


def tensor_train_contraction(states_tensor, cores):
    #print("input:", states_tensor.name, states_tensor.get_shape().as_list())
    #print("mat_ranks", mat_ranks)
    #print("mat_ps", mat_ps)
    #print("mat_dims", mat_dims)
    #print("mat_size", mat_size)

    abc = "abcdefghij"
    klm = "klmnopqrstuvwxy"

    def _get_indices(r):
        indices = "%s%s%s" % (abc[r], klm[r], abc[r+1])
        return indices

    def _get_einsum(i, s2):
        #
        s1 = _get_indices(i)
        _s1 = s1.replace(s1[1], "")
        _s2 = s2.replace(s2[1], "")
        _s3 = _s2 + _s1
        _s3 = _s3[:-3] + _s3[-1:]
        s3 = s1 + "," + s2 + "->" + _s3
        return s3, _s3

    n = len(cores)
    # first factor
    x = "z" + klm[:n] # "z" is the batch dimension
    
    # print(mat_core.get_shape().as_list())

    _s3 = x[:1] + x[2:] + "ab"
    einsum = "akb," + x + "->" + _s3
    x = _s3

    #print("einsum", einsum, cores[0].get_shape().as_list(), states_tensor.get_shape().as_list())

    out_h = tf.einsum(einsum, cores[0], states_tensor)
    #print(out_h.name, out_h.get_shape().as_list())

    # 2nd - penultimate latent factor
    for i in range(1, n):

        # We now compute the tensor inner product W * H, where W is decomposed
        # into a tensor-train with D factors A^i. Each factor A^i is a 3-tensor,
        # with dimensions [mat_rank[i], hidden_size, mat_rank[i+1] ]
        # The lag index, indexing the components of the state vector H,
        # runs from 1 <= i < K.

        # print mat_core.get_shape().as_list()

        einsum, x = ss, _s3 = _get_einsum(i, x)

        #print("order", i, ss)

        out_h = tf.einsum(einsum, cores[i], out_h)
        #print(out_h.name, out_h.get_shape().as_list())

    # print "Squeeze out the dimension-1 dummy dim (first dim of 1st latent factor)"
    out_h = tf.squeeze(out_h, [1])
    return out_h

    
    
    
    
    
    
    
    
#def _poly_expand(h, constant_permute_tensors, batch_size, poly_order):
#    # print(constant_sparse_tensors[0])
#    h_pow_series =[]
#    h_pow = h
#    for i in range(poly_order-1):
#        h_pow = _outer_product(batch_size, h_pow, h)
#        h_pow_series.append(h_pow)
#    
#    def contract(st, h):
#        len_st = len(st.get_shape().as_list())
##        print("st and h :", st, h, len_st)
#        return tf.transpose(sparse_tensor_dense_tensordot(st, h, [[i for i in range(len_st-1)],[j+1 for j in range(len_st-1)]]))
#    
#    return tf.concat([h] + [contract(constant_permute_tensors[k], h_pow_series[k]) for k in range(poly_order-1)], 1)

#def _poly_expand(h, permute_series_list, batch_size, poly_order):
#    # print(constant_sparse_tensors[0])
#    res = h
#    h_pow = h
#    for order in range(1,poly_order):
#        h_pow = _outer_product(batch_size, h_pow, h)
#        #print(h_pow)
#        r=tf.gather_nd(tf.transpose(h_pow),permute_series_list[order])
#        #print("r",r)
#        res = tf.concat([res,tf.transpose(r)],1)       
#    return res
    
def _poly_expand(h, permute_series_list, batch_size, poly_order):
    # print(constant_sparse_tensors[0])
    for order in range(poly_order):
        if order == 0:
            res = h
        else:
            res = tf.concat([res, h**poly_order], 1)
        #print(res)
    return res
    
def mera(states, weights, rank_vals):
    s = states
    index = 0
    for i in range(len(rank_vals)-2):
        length = rank_vals[i]*rank_vals[i]*rank_vals[i+1]*rank_vals[i+1]
        core_u = tf.slice(weights, [index], [length])
        core_u = tf.reshape(core_u, [rank_vals[i],rank_vals[i],rank_vals[i+1],rank_vals[i+1]])
        index=index+length

        length = rank_vals[i+1]*rank_vals[i+1]*rank_vals[i+1]
        core_w = tf.slice(weights, [index], [length])
        core_w = tf.reshape(core_w, [rank_vals[i+1],rank_vals[i+1],rank_vals[i+1]])
        index=index+length

        #iterator = np.random.randint(len(tf.shape(s))-1)
        iterator = len(s.get_shape())-2
        s = tf.tensordot(s, core_u, [[iterator,iterator+1],[0,1]])
        for _ in range((len(s.get_shape())-1)//2-1):
            iterator = iterator - 2
            s = tf.tensordot(s, core_u, [[iterator,iterator+1],[0,1]])
            s = tf.tensordot(s, core_w, [[-4, -1],[0,1]])
        s = tf.tensordot(s, core_w, [[iterator, -2],[0,1]])
        #print(iterator)
        #print(list(range(0,iterator))+list(range(iterator,len(s.get_shape())-1+iterator))[::-1])
        s = tf.transpose(s, list(range(0,iterator))+list(range(iterator,len(s.get_shape())-1+iterator))[::-1])
        #print(s)
        
    shape = s.get_shape().as_list()[1:]
    len_shape = len(shape)
    core_t = tf.slice(weights, [index], [-1])
    core_t = tf.reshape(core_t, shape+[rank_vals[-1]])
    s = tf.tensordot(s, core_t, [list(range(1,len_shape+1)),list(range(len_shape))])
    return s




def vicissitude(inputs, states, permute_series_list, hidden_size, expanded_hidden_size, poly_order, rank_vals, output_size, bias, bias_start=0.0):

    num_lags = len(states)#alpha_1 to alpha_{K-1}

    batch_size = tf.shape(inputs)[0] 
    input_size= inputs.get_shape()[1].value
    #total_hidden_size = (hidden_size * len(states) + 1 )
    total_hidden_size = expanded_hidden_size + 1
    #print("states", states)
    
#    mat_dims = np.ones((num_lags,)) * total_hidden_size
#    mat_ranks = np.concatenate(([1], [2,4,3,5,4,6,8], [output_size]))
#    mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * mat_dims * mat_ranks[1:])),dtype=np.int32)
#    mat_size = mat_ps[-1]
#    mat = vs.get_variable("weights_h", mat_size) # h_z x h_z... x output_size
#    states_vector = tf.concat(states, 1)
#    states_vector = tf.concat( [states_vector, tf.ones([batch_size, 1])], 1)
#    states_tensor = states_vector
#    for order in range(num_lags-1):
#        states_tensor = _outer_product(batch_size, states_tensor, states_vector)
#    cores = []
#    for i in range(num_lags):
#        # Fetch the weights of factor A^i from our big serialized variable weights_h.
#        mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
#        mat_core = tf.reshape(mat_core, [mat_ranks[i], total_hidden_size, mat_ranks[i + 1]])   
#        cores.append(mat_core)
#    #print(cores)    
#    out_h = tensor_train_contraction(states_tensor, cores)
#    

        
    #st_test = list(map(lambda h: _poly_expand(h, permute_series_list, batch_size, poly_order), states))
    #print("st_test", st_test)    
    st_test =states
    for order in range(num_lags):
        states_vector = st_test[order]
        states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
        if order == 0:
            states_tensor = states_vector
        else:
            states_tensor = _outer_product(batch_size, states_tensor, states_vector)
    #print("states tensor", states_tensor)
    rank_vals = rank_vals+[output_size]
    w_size = 0
    for i in range(len(rank_vals)-2):
        w_size = w_size + rank_vals[i]*rank_vals[i]*rank_vals[i+1]*rank_vals[i+1]
        w_size = w_size + rank_vals[i+1]*rank_vals[i+1]*rank_vals[i+1]
    w_size = w_size + rank_vals[-2]*rank_vals[-2]*rank_vals[-1]
    #w_size=3*3*6*6+6*6*6+6*6*4*4+4*4*4+4*4*output_size
    weights = vs.get_variable("w_h", w_size)
    out_h = mera(states_tensor, weights, rank_vals)


    #print("out_h", out_h)
    # Compute h_t = U*x_t + W*H_{t-1}
    
    weights_x = vs.get_variable("weights_x", [input_size, output_size] )
    out_x = tf.matmul(inputs, weights_x)


    res = tf.add(out_x, out_h)

    # print "END OF CELL CONSTRUCTION"
    # print "========================"
    # print ""

    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])

    return nn_ops.bias_add(res,biases)
