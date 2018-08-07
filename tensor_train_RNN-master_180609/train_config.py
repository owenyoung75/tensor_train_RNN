import numpy as np
import tensorflow as tf

from math import factorial
class TrainConfig(object):

    def __init__(self):

        def recur(permute_series_list):
            permute_series_new = []
            for series_i in permute_series_list[-1]:
                for k in range(series_i[-1], self.hidden_size):
                    s=list(series_i)
                    s.append(k)
                    permute_series_new.append(s)
            l=list(permute_series_list)
            l.append(permute_series_new)
            return l
        def build_permute_tensor(permute_series, hidden_size, order):
            len_ps = len(permute_series)
            concat = np.concatenate((permute_series, list(map(list, zip(*[[x for x in range(len_ps)]])))), axis=1)
            shape = [hidden_size]*order
            shape.append(len_ps)
            #  print("shape", shape)
            return tf.SparseTensor(concat, [1.0]*len_ps, shape)



        self.init_scale = 0.1
        self.learning_rate = 1e-2
        self.decay_rate = 0.8
        self.max_grad_norm = 10

        self.num_freq = 2
        self.training_steps = int(5e1)
        self.keep_prob = 1.0 # dropout
        self.sample_prob = 0.0 # sample ground true
        self.batch_size = 50

        self.hidden_size = 2 # dim of h
        self.poly_order = 2
            
        self.permute_series_list =[[[i] for i in range(self.hidden_size)]]
        for _ in range(self.poly_order-1):
            self.permute_series_list = recur(self.permute_series_list)
            
        self.expanded_hidden_size = sum([factorial(i+1+self.hidden_size-1)//factorial(i+1)//factorial(self.hidden_size-1) for i in range(self.poly_order)])

        #self.constant_permute_tensors = [build_permute_tensor(self.permute_series_list[i], self.hidden_size, i+1) for i in range(1, self.poly_order)]
     
        self.num_layers = 1
        self.inp_steps = 20 
        self.horizon = 1

        self.num_lags = 8 # tensor prod order
        #8,4,2,1
        self.rank_vals= [self.hidden_size+1]+[6, 4]
