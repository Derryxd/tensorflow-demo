# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:10:38 2019
SimpleNet built for linear regression
@author: ldz
"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
'''Layer'''
# =============================================================================
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s'% n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/Weights',weights)
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name+'/Biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/Outputs', outputs)
        return outputs
    
# =============================================================================
'''Train and Test'''
# =============================================================================
#Inputs
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise 

#define placeholder for inpputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#add hidden layer
L1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

#add output layer
prediction = add_layer(L1, 10, 1, n_layer=2, activation_function=None)

#the error between prediciton and real data
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    tf.summary.scalar('Loss', loss)

#Optimizer to train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#Intialization
init = tf.global_variables_initializer()

#sess = tf.Session() #... ... sess.close()
with tf.Session() as sess:
    #Tensorboard : tensorboard --logdir='logs/'
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs",sess.graph)
    
    #Intialization through Session
    sess.run(init)
    
    #Visualization
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    
    #train the network
    for i in range(1001):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i%50 == 0:
            #Plot in PyPt5
            print("loss")
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    #        print("prediction")
    #        print(sess.run(prediction, feed_dict={xs: x_data}))
            try:
                ax.lines.remove(lines[0])  
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
            
            #Plot in tensorborad
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)
            
# =============================================================================
#修改C:\...\Python\Python36\site-packages\tensorboard\manager.py的51行内容：
#     # serialize=lambda dt: int(
#     #     (dt - datetime.datetime.fromtimestamp(0)).total_seconds()),
#     #-------------------------------------------------------------
#     #以上serialize..语句存在bug
#     #运行tensorboard时报错：  OSError: [Errno 22] Invalid argument
#     #以下做修改。
#     #-------------------------------------------------------------
#     serialize=lambda dt: int(dt.strftime("%S")), #一定要写大写S
# =============================================================================
