# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:31:24 2019

@author: Dell
"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

class TrainConfig:
    BATCH_START = 0
    TIME_STEPS = 20
    BATCH_SIZE = 50
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    CELL_SIZE = 10
    LR = 0.006

class TestConfig(TrainConfig):
    TIME_STEPS = 1

def get_batch(config):
    # xs(50batch, 20steps)
    x = np.arange(config.BATCH_START, 
                  config.BATCH_START + config.TIME_STEPS*config.BATCH_SIZE) \
                  .reshape((config.BATCH_SIZE, config.TIME_STEPS)) / (10*np.pi)
    seq = np.sin(x)
    res = np.cos(x)
    config.BATCH_START += config.TIME_STEPS
#    plt.plot(xs[0,:], res[0,:], seq[0,:], 'b--')
#    plt.show()
    ## return seq and res with shape (batch, step, input)
    return [seq[:,:,np.newaxis], res[:,:,np.newaxis], x]

class LSTMRNN(object):
    def __init__(self, config):
        self._n_steps = config.TIME_STEPS
        self._input_size = config.INPUT_SIZE
        self._output_size = config.OUTPUT_SIZE
        self._cell_size = config.CELL_SIZE
        self._batch_size = config.BATCH_SIZE
        self._lr = config.LR
        self._built_RNN()
    
    def _built_RNN(self):
        with tf.variable_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None, self._n_steps, self._input_size], name='xs')
            self.ys = tf.placeholder(tf.float32,[None, self._n_steps, self._input_size], name='ys')         
        with tf.name_scope('RNN'):
            with tf.variable_scope('in_hidden'):
                self.add_input_layer()
            with tf.variable_scope('LSTM_cell'):
                self.add_cell()
            with tf.variable_scope('out_hidden'):
                self.add_output_layer()
        with tf.name_scope('cost'):    
            self.compute_cost()
        with tf.variable_scope('train'):
            self._lr = tf.convert_to_tensor(self._lr)
            self.train_op = tf.train.AdamOptimizer(self._lr).minimize(self.cost)
            
    def add_input_layer(self):
        # L_in_x (batch*n_step, in_size)
        L_in_x = tf.reshape(self.xs, [-1, self._input_size], name='2_2D') 
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self._input_size, self._cell_size])
        # Bs (in_size, cell_size)
        Bs_in = self._bias_variable([self._cell_size, ])
        # L_in_y (batch*n_step, cell_size)
        with tf.name_scope('Wx_plus_b'):
            L_in_y = tf.matmul(L_in_x, Ws_in) + Bs_in 
            self.L_in_y = tf.reshape(L_in_y, [-1, self._n_steps, self._cell_size], name='2_3D')
            
    def add_cell(self):
#        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self._batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.L_in_y, initial_state=self.cell_init_state, time_major=False)
    
    def add_output_layer(self):
        # L_out_x (batch*n_step, cell_size)
        L_out_x = tf.reshape(self.cell_outputs, [-1, self._cell_size], name='2_2D') 
        # Ws (cell_size, output_size)
        Ws_out = self._weight_variable([self._cell_size, self._output_size])
        # Bs (output_size)
        Bs_out = self._bias_variable([self._output_size, ])
        # L_out_y (batch*n_step, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(L_out_x, Ws_out) + Bs_out
            
    def compute_cost(self):
        # Loss for every step
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(self.pred,[-1], name='reshape_pred')],
                [tf.reshape(self.ys, [-1], name='reshape_target')],
                [tf.ones([self._batch_size * self._n_steps], dtype=tf.float32)],
                average_across_timesteps = True,
                softmax_loss_function = self.ms_error,
                name = 'losses')
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                    tf.reduce_sum(losses, name='losses_sum'),
                    tf.cast(self._batch_size, tf.float32), 
                    name='average_cost')
            tf.summary.scalar('cost',self.cost)
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))
    ## Equivalent to following with tf_y's shape [None, TIME_STEP, INPUT_SIZE]:
#    cost = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
    
    @staticmethod        
    def _weight_variable(shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
    
    @staticmethod
    def _bias_variable(shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
    
if __name__ == '__main__':
    
    tf.reset_default_graph()
    train_config = TrainConfig()
    test_config = TestConfig()
    
    # model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    with tf.variable_scope('rnn_run') as scope:
        sess = tf.Session()
        
        # TrainNet
        train_rnn = LSTMRNN(train_config)
#        merged = tf.summary.merge_all()
#        writer = tf.summary.FileWriter("logs", sess.graph)
        ## relocat the local dir and run this line to view it on Chorme (http://0.0.0.0:6006/):
        ## $ tensorboard --logdir = 'logs'
        
        # TestNet with some parameter changed compared to trainNet         
        scope.reuse_variables()
        test_rnn = LSTMRNN(test_config)
        
        # Initial the variable in Nets
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        
        plt.figure('rnn')
        plt.ion()
        plt.show()
        for i in range(200):
            seq, res, xs = get_batch(train_config)
            if i == 0:
                feed_dict = {
                        train_rnn.xs: seq,
                        train_rnn.ys: res
                        # create initial state
                        }
            else:
                feed_dict = {
                        train_rnn.xs: seq,
                        train_rnn.ys: res,
                        train_rnn.cell_init_state: state
                        ## use last state as the initial state for this run
                        }
            _, cost, state, pred = sess.run(
                    [train_rnn.train_op, 
                     train_rnn.cost,
                     train_rnn.cell_final_state,
                     train_rnn.pred],
                     feed_dict=feed_dict)
            
            # Plot
            plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:train_config.TIME_STEPS], 'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.05)
            
            if i % 20 ==0:
                print('cost: ', round(cost, 4))
#                result = sess.run(merged, feed_dict)
#                writer.add_summary(result, i)

        sess.close()
            
    
    
    
    
    
    
    
    