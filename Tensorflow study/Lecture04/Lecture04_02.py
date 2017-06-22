import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Lab 4-2
# # Loading data from file
# import numpy as np
# # 단점이 같은 datatype 이어야해서!!
# xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
# # xy로 이 전체를 array로 읽어오게 된다!!
# x_data = xy[:, 0:-1] # 행 : 전체, 열 : 처음부터 마지막 1개 빼고
# y_data = xy[:, [-1]] # 행 : 전체, 열 : 마지막 한개만
# # Make sure the shpae and data are OK
# print(x_data.shape, x_data, len(x_data)) # 6행 3열
# print(y_data.shape, y_data) #6행 1열
# # placeholders for a tensor that will be always fed.
# X = tf.placeholder(tf.float32, shape=[None, 3])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([3, 1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
# # Hypothesis
# hypothesis = tf.matmul(X, W) + b # Matrix Multiply
# # Simplified cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
# # Minimize
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)
# # Launch the graph in a session
# sess = tf.Session()
# # Initializes global variables in the graph
# sess.run(tf.global_variables_initializer())
# # Set up feed dict variables inside the loop.
# for step in range(2001):
#     cost_val, hy_val, _ = sess.run(
#         [cost, hypothesis, train],
#         feed_dict={X: x_data, Y: y_data})
#     if step % 10 == 0:
#         print(step, "Cost: ", cost_val,
#               "\nPrediction:\n", hy_val)
# # Ask my score
# print("Your score will be ", sess.run(hypothesis,
#                              feed_dict={X: [[100, 70, 101]]}))
# print("Other scores will be ", sess.run(hypothesis,
#                              feed_dict={X: [[60,70,110],[90,100,80]]}))



# 파일이 너무 커서 메모리가 부족한 경우가 있다. 그래서  tensorflow 에서는 Queue Runners 를 제공해 준다.
# 여러 file을 queue에 쌓게되고,,,,
# queue 에서 조금씩 빼서 쓰면 된다!!
# 1. 가지고 있는 파일들을 list를 만들어 준다.
# filename_queue = tf.train.string_input_producer(
#     ['data-01-test-score.csv', 'data-02-test-score.csv', ...],
#     shuffle=False, name='filename_queue')
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'],
    shuffle=False, name='filename_queue') # queue를 만든다

# 2. 파일을 읽어 올릴 reader 를 정의해 준다.
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 3. value 값을 어떻게 파싱할 것인가?! 이해할 것인가?! 를 decode_csv 라는 것으로 가져올 수 있다.
record_defaults = [[0.],[0.],[0.],[0.]] # 각 열이 어떤 data type 인지를 정의해 줄 수 있다.
xy = tf.decode_csv(value, record_defaults=record_defaults) # csv로 decode하라!!

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) # 읽어온 데이터를 x와 y로 나누고 각각 batch해 줘라 한번에 10개씩!!

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch]) # 펌프를 해서 값을 빼오는 거다!!
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_batch, Y:y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val,
              "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)












