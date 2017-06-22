import tensorflow as tf

# Training and Test datasets
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Q.axis = 1이 무엇을 뜻하나?!
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1) # Q. 여기서 1 은 뭐지?!
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   for step in range(201):
       cost_val, W_val, _ = sess.run([cost, W, optimizer],
                       feed_dict={X: x_data, Y: y_data})
       print(step, cost_val, W_val)

   # 위는 최적 모델을 구현하는 거고.

   # predict, 예측값 찾고!!
   print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))

   # Calculate the accuracy, 예측률 계산
   print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


# Big learning rate는 NaN 에러를 발생 시킬 수 있다.
# Small learning rate는 Minimum Cost 를 찾는데 비효율적일 수 있다.
# Non-normalized inputs 또한 Minimum Cost 를 찾는데 문제를 일으킬 수 있다.
    # 그래서 표준화는 해준다.
    # xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
    #               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
    #               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
    #               [816, 820.958984, 1008100, 815.48999, 819.23999],
    #               [819.359985, 823, 1188100, 818.469971, 818.97998],
    #               [819, 823, 1198100, 816, 820.450012],
    #               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
    #               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])
    # xy = MinMaxScaler(xy)
    # print(xy)






