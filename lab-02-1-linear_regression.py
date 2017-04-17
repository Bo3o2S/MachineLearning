# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find values for W and b to compute y_data = x_data * W +b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loass function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# Learns best fit W:[ 1.], b:[ 0.]