# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:01:04 2019

@author: ldz
"""

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os

# =============================================================================
'''matrix multiply np.dot(m1,m2)'''
# =============================================================================
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2) 
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
#    sess.close()

# =============================================================================
'''Variable'''
# =============================================================================
state = tf.Variable(0, name = 'counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# =============================================================================
'''Placeholder'''
# =============================================================================
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
    
# =============================================================================
'''Layer and Net'''    
# =============================================================================
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

# =============================================================================
''' Save & read to file '''
# =============================================================================
## Save
#W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
#b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
#init = tf.global_variables_initializer()  # notice!!
#
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, "my_net/save_net.ckpt")
#    print("Save to path:", save_path)

## Restore
#Wr = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
#br = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
#
#restore = tf.train.Saver()
#with tf.Session() as sess:
#    restore.restore(sess, "my_net/save_net.ckpt")
#    print("weights:", Wr)
#    print("biases:", br)
#
## Check
#model_dir = r'E:\test'
#checkpoint_path = os.path.join(model_dir, "model.ckpt")
#reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#    print("tensor_name: ", key)
#    #print(reader.get_tensor(key))

# =============================================================================
''' name_scope & variable_scope '''
# =============================================================================
tf.reset_default_graph()

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)
    var1 = tf.get_variable(name='var11', shape=[1], dtype=tf.float32, initializer=initializer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)        # var1:0 ->> var11:0
    print(sess.run(var1))   # [ 1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]

with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3',)

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]
    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse)) # [ 4.]
    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse)) # [ 3.]
    
    
    
    