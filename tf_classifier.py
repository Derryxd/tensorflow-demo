# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:08:35 2019
Classifier 
@author: ldz
"""
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, layer_name, 
              keep_prob=1, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs) #用于tensorboard
        return outputs
    
# =============================================================================
''' mnist example '''
# =============================================================================
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_pre = tf.equal(tf.arg_max(y_pre,1), tf.arg_max(v_ys,1))  #0：按列计算，1：行计算
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 784=28*28 per picture
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10, 'Prection', activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1])) #loss
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

sess.close()  

# =============================================================================
''' sklearn example & overfitting '''
# =============================================================================
#load data
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xr = tf.placeholder(tf.float32, [None, 64])  # 8*8
yr = tf.placeholder(tf.float32, [None, 10])

# add output layer
L1 = add_layer(xr, 64, 100, 'L1', activation_function=tf.nn.tanh)
predict = add_layer(L1, 100, 10, 'L2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(yr*tf.log(predict),
                                              reduction_indices=[1])) #loss
tf.summary.scalar('Loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess=tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={xr:x_train, yr:y_train, keep_prob:0.6})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xr:x_train, yr:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xr:x_test, yr:y_test, keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
        
sess.close()  


