import tensorflow as tf
print tf.__version__

hello = tf.constant("Hello, TensorFlow!")

sess = tf.Session()

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
node4 = tf.constant([[[1., 2., 3.]], [[7., 8., 9.]]])

print("node1: ", node1)
print("node2: ", node2)
print("node3: ", node3)
print("node4: ", node4)
