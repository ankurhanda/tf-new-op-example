#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: Ankur Handa
"""

import unittest
import numpy as np
import tensorflow as tf

# https://stackoverflow.com/questions/44403127/adding-a-gpu-op-in-tensorflow

with tf.Session('') as sess:
   	
  x = tf.placeholder(tf.float32, shape = (1, 6))

  x_trans, x_rot = tf.split(x, 2, 1)

  col_1 = tf.cross(x_rot, tf.constant([1, 0, 0], shape=(1, 3), dtype=tf.float32))
  col_2 = tf.cross(x_rot, tf.constant([0, 1, 0], shape=(1, 3), dtype=tf.float32))
  col_3 = tf.cross(x_rot, tf.constant([0, 0, 1], shape=(1, 3), dtype=tf.float32))

  omega = tf.concat([col_1, col_2, col_3],axis=0)
  omega_t = tf.transpose(omega)

  theta = tf.sqrt(tf.reduce_sum(tf.square(x_rot)))
  sin_theta = tf.sin(theta)

  sin_t_by_t = tf.div(sin_theta, theta)
  one_min_cos_t_by_t = tf.div(tf.add(1.0, -tf.cos(theta)), tf.square(theta))

  R = tf.scalar_mul(sin_t_by_t,omega_t) + \
      tf.scalar_mul(one_min_cos_t_by_t, tf.matmul(omega_t, omega_t)) + \
      tf.eye(3,dtype=tf.float32)

  theta_minus_sin_theta_div_theta3 = tf.div(tf.add(theta, -sin_theta), tf.pow(theta,3))

  V = tf.scalar_mul(one_min_cos_t_by_t, omega_t) +\
      tf.scalar_mul(theta_minus_sin_theta_div_theta3, tf.matmul(omega_t, omega_t)) \
      + tf.eye(3,dtype=tf.float32)

  Vu = V * x_trans

  for i in range(1):
    cur_x = np.random.rand(1,6)
    print('cur_x ..') 	  
    print(cur_x)
    gradient_tf = sess.run(omega, feed_dict = {x: cur_x})
    
    print('out ..')
    print(gradient_tf)
    print(sess.run(omega_t, feed_dict={x:cur_x}))

    print(sess.run(tf.eye(3,dtype=tf.float32)))
    print('theta = ', sess.run(theta, feed_dict={x:cur_x}))

    print('R = ')
    rMat = sess.run(R, feed_dict={x:cur_x})
    print(rMat)

    print(np.linalg.det(rMat))

    print('Vu = ', sess.run(Vu, feed_dict={x:cur_x}))


