# import tensorflow as tf
# tf.__version__

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sess = tf.Session()
hello = tf.constant('Hello, TensorFlow')
print(sess.run(hello))
print(str(sess.run(hello), encoding='utf-8'))   # unicode --> utf-8


hello = tf.constant("Hello, TensorFlow")

sess = tf.Session()

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
print(node1)

node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1:", node1, "node2", node2)
print("node3: ", node3)

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

# (1) Build graph (tensors) using TensorFlow operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# (2) feed data and run graph (operation) sees.run(op)
# (3) update variables in the graph(and return values)
sess = tf.Session()
print("sess.run(node1, node2) :", sess.run([node1, node2]))
print("sess,run(node3) :", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
# 걍 덧셈임!
# placeholder는 모델을 계속 update 할 수 있다

# Everything is Tensor
# 3 # a rank 0 tensor; this is a scalar with shape []
# [1. , 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
# [[1.,2.,3.],[4.,5.,6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1.,2.,3.]],[[7.,8.,9.]]]
t = tf.constant([1., 2., 3.])
print(t)

# rank로 차원을 설명한다. rank=1 은 vector... 2는 matrix 등등..
# type은 DT_FLOAT(tf.float32), DT_DOUBLE(tf.float64),
# DT_INT8(tf.int8), DT_INT16(tf.int16), DT_INT32(tf.int32),
# DT_INT64(tf.int64)
