import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32) #xor data set
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) # xor result

X = tf.placeholder(tf.float32, [None, 2]) #set as placeholder
Y = tf.placeholder(tf.float32, [None, 1]) #set as placeholder

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)) #logistic classification cost function

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) #traning

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # if result is upper than 0.5, predicted is 1, else predicted is 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) #accuracy

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initiallize for use variables

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y:y_data}) #
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y:y_data})
    print("Hypothesis: ", h, "Correct : ", c, "Accuracy: ", a)