#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: Ankur Handa
"""

import unittest
import numpy as np
import tensorflow as tf
#import _inner_product_grad
#inner_product_module = tf.load_op_library('build/libinner_product.so')
#import _inner_product_grad
se3_transform_module = tf.load_op_library('build/libse3_layer.so')

with tf.Session('') as sess:
   	
  x = tf.placeholder(tf.float32, shape = (3,3,3))
  T = tf.placeholder(tf.float32, shape = (1,3,4))
  
  out = se3_transform_module.se3_transform(x, T)          

            
  for i in range(1):
    cur_x = np.random.rand(3,3,3)
    cur_T = np.random.rand(1,3,4)

    print('cur_x ..') 	  
    print(cur_x)
    print('cur_T ..')
    print(cur_T) 
    gradient_tf = sess.run(out, feed_dict = {x: cur_x, T: cur_T})
    
    print('out ..')
    print(gradient_tf)
