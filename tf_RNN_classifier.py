# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:06:13 2019

@author: Dell
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128
n_input = 28
n_steps = 28
n_hidden_units = 128 # neurons in hidden layer
n_classes = 10

# define placeholder for inputs to network 
x = tf.placeholder(tf.float32, [None, n_steps, n_input])  # 28*28
y = tf.placeholder(tf.float32, [None, n_classes])

# define weights 
weights = {
        #(28, 128)
        'in' : tf.Variable(tf.random_normal([n_input, n_hidden_units])),
        #(10,)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
biases = {
        #(128,)
        'in' : tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])) ,
        #(10,)
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
        }

# many to one : classifier
def RNN(X, weights, biases):
    #One: hidden layer for input to cell
    X = tf.reshape(X,[-1, n_input]) # X(128 batch, 28 steps, 28 inputs) ->> (128*28, 28 inputs)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    #Two: cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # state(batch_size * n_hidden_units)
    # outputs is a list for every step, and states is a tuple for the final result
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    
    #Three: hidden layer for outputs as the final results
    # take final result(m_state) as prediction
    results = tf.matmul(states[1], weights['out']) + biases['out']
    # Or:
#    #unpack(unstack) to list [(batch, outputs)..]*steps
#    outputs = tf.unstack(tf.transpose(outputs,[1,0,2])) # states is the last outputs (n_step, batchsize, output_size)
#    # the same as following:  outputs = tf.unstack(outputs,axis=1) 
#    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    return results
    
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x:v_xs})
    correct_pre = tf.equal(tf.math.argmax(y_pre,1), tf.arg_max(v_ys,1))  #0：按列计算，1：行计算
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={x:v_xs, y:v_ys})
    return result

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_operation = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.math.argmax(pred,1), tf.math.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
        sess.run(train_operation, feed_dict={x:batch_xs, y:batch_ys})
        if step %20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step += 1
            
   