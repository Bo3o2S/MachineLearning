#Lab 9 XOR

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None,2], name='x-input')
Y = tf.placeholder(tf.float32, [None,1], name='y-input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram('weights1', W1)
    b1_hist = tf.summary.histogram('biases1', b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram('weights2', W2)
    w2_hist = tf.summary.histogram('biases2', b2)
    layer2_hist = tf.summary.histogram("hypothesis", hypothesis)

# cost/loss function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))
    cost_sum = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:

