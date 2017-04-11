#it's get from
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0],[1,1]], dtype=np.float32) #input data X
y_data = np.array([[0], [1], [1], [0]]) #input data Y

X = tf.placeholder(tf.float32, [None, 2]) #placeholder Vector X
Y = tf.placeholder(tf.float32, [None, 1]) #placeholder Vector Y

#first layer in NN
#In first layer, Operate sigmoid
W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

#second layer in NN
#In second layer, get result of Operating first layer and also take sigmoid function as operation
#And second layer, Operation is hypothesis
W2 = tf.Variable(tf.random_normal([2,2]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

#logistic classification cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

#training cost function
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#predicte it's 1 or 0. if hypothesis is upper than 0.5, it's True, else it's False.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

#predict accracy about Y. if predicted variable equal Y, it's return 1. else it's return 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
# initialize variable
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        #run train for caculating minimum cost
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0: #print step in 100
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))
    #print hypothesis, predicted, accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
