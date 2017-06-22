import tensorflow as tf

    # Reading data and set variables
# MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist)

print (mnist.train.labels[1])
print (mnist.train.images[1])

# # 그림으로 그려보면.
# import numpy as np
#
# arr = np.array(mnist.train.images[1])
# arr.shape = (28,28)
#
# # %matplotlib inline
# import matplotlib.pyplot as plt
# plt.imshow(arr)
# plt.show()



nb_classes = 10
# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))


    # Softmax!
# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


    # Training epoch/batch
# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print("Learning finished")


# Training epoch / batch
# In the neural network terminology:
# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass.
# The higher the batch size, the more memory space you will need.
# number of iterations = number of passes, each pass using [batch size] number of examples.
# To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
# Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

# 4. epoch
# 훈련용 사진 전체를 딱 한 번 사용했을 때 한 세대(이폭, epoch)이 지나갔다고 말합니다.
# cifar10의 경우 사진 60,000장 중 50,000장이 훈련용, 10,000장이 검사용으로 지정되어 있습니다.
# 그런데 max_iter에서 훈련에 사진 6,000,000장을 사용하기로 했기 때문에 50,000장의 훈련용 사진이
# 여러번 재사용되게 됩니다. 정확히 계산해보면 6,000,000 / 50,000 = 120 이니 한 사진이 120번 씩 재사용될 것입니다.
# 이 경우 120 세대(epoch)라고 말합니다. 검사용의 경우 사진 10,000장을 사용하기로 했는데 실제로도
# 사진이 10,000장 있으니 딱 한 세대만 있는 것입니다.

# 1. batch_size
# 배치(batch)는 한 번에 처리하는 사진의 장 수를 말합니다.
# Caffe에서 기본으로 제공되는 cifar10 예제의 cifar10_full_train_test.prototxt 파일을
# 열어보면 batch_size: 100 이라는 부분이 있습니다.
# 한 번에 100장의 사진을 처리한다는 의미입니다.


    # Report results on test dataset
# Test the model using test sets
print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
# 이부분 에러 발생하네... 이유가 뭐지?!


# Sample image show and prediction
import matplotlib.pyplot as plt
import random

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:", sess.run(tf.argmax(hypothesis, 1),
                              feed_dict={X: mnist.test.images[r:r+1]}))
plt.imshow(mnist.test.images[r:r+1].
           reshape(28,28), cmap="Greys", interpolation='nearest')
plt.show()




################### 다른 방식 예제 실시 ###############
#     # Reading data and set variables
# # MNIST Dataset
# from tensorflow.examples.tutorials.mnist import input_data
# # Check out https://www.tensorflow.org/get_started/mnist/beginners for
# # more information about the mnist dataset
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# print (mnist.train.labels[1])
# print (mnist.train.images[1])
#
# import tensorflow as tf
# import numpy as np
#
# arr = np.array(mnist.train.images[1])
# arr.shape = (28,28)
#
# import matplotlib.pyplot as plt
# plt.imshow(arr)
# plt.show()
#
# # 이제 train 을 해보면
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# # input은 총 55000개의 784 pixel 을 가진 학습 데이터가 사용되며, output은 10개(0~9, 숫자)의
# # classification을 가진 55000개의 결과가 만들어 질 것이다.
#
# # 우리는 이제 tensorflow 연산 시 데이터를 tensorflow 에게 보내기 위한 공간을 만들 필요가 있다.
# # placeholder라고 하는 함수를 이용하자. [None, 784]는 행 길이는 제한을 두지 않겠다는 것을 의미한다.
# # W 는 784 x 10 차원의 0 값을 가진 행렬로 정의하자. None x 784 행렬을 10개의 class로 분류하기 위해 W는
# # 10 차원의 행렬이 되어야 한다. 0은 초기값이며 학습을 통해 그 값을 계속 변경해 나갈 것이다.
# # b도 10차원의 0 값을 가진 행렬로 정의해 두자. b는 bias 의 줄임 표현으로 W와
# # 입력값의 결과에 추가적인 정보를 더하기 위한 값을 의마한다.
#
# # y값을 계산하는 것을 보면 tf.matmul(x.W) + b 라고 적혀 있는데 이것은 단순하게
# # 행렬 곱을 의마하며 방정식으로 말하자면 Wx+b 를 의미한다.
# # 이 결과 값에 대해 softmax라는 함수를 취하는데 해당 결과 값에 대해서 softmax를 위하게 되면
# # 확률값으로 변하게 된다.
# # 위의 8이란 필기체 사진을 예로 들어보면 위의 사진을 80% 가량 8이라고 인식할 수 있지만 10%가량은 9라고 인식할 수 있고
# # 그 나머지는 나머지 숫자들로 인식할 수가 있다.
# # Wx 는 해당 숫자가 무엇을 나타내는지 그 증거를 찾는 과정이라고 할 수 있으며
# # b는 추가적인 증거를 더하는 과정이라고 생각하면 된다.
#
# # 위의 그림에서 색이 푸른색으로 나타나는 지점은 W를 마이너스를 줌으로써 해당 증거 결과 값을 음수 값을 띄게 하고,
# # 숫자 부분을  W를 플러스 값을 줌으로써 해당 증거 값을 양수 값을 띄게 한다.
# # 이렇게 나타난 증거 결과에 대해서 softmax 값을 취함으로써 확률로 변환시켜 주는 것이다.
# #
# # y_ 는 tensorflow로 부터 결과 값을 받아오기 위한 placeholder를 정의한 것이다.
# #
# # 그 후에 loss 함수를 정의해 주는데, cross_entropy라  불리우는 방식을 사용한다.
# # 기존 RMSE와 그 의미 및 목표는 같다고 볼 수 가 있다. 다른점은 cross_entropy는 확률 분포에
# # 대한 차이점을 나타내는 것이라고 하겠다.
# #
# # 우리가 one_hot encode로 표현한 확률 분포와 실제 계산해서 나온 활률 분포 간의 차이를 구해서 그 값이 가장 작은 지점에서의
# # weight 값을 찾아내는 것이다.
# #
# # loss 함수 까지 정의가 끝났으면 이제  gradient descent optimizer에 learning rate와 loss 함수를
# # 등록해 주면 사전 작업은 모두 끝났다.
# #
# # 이제 training을 돌리기 전에 tf의 모든 변수들을 초기화ㅑ 시켜준다. tensor flow는 lazy evaluation 방식이라
# # 코드가 작성되면 바로 실행되는 것이 아니라 session.run이 실행되어야 실제 함수가 동작한다. 세션을 선언한 후
# # session.run에 정의한 init 함수를 집어 넣자.
# #
# # 이제 for문을 1000번을 돌려, 100개 씩 input_images 데이터와 input_labels 데이터를 가져온다. session.run을
# # 실행하여 아까 정의한 training 함수를 실행시키면 모든 training이 완료된다.
# #
# # 이제 만들어진 model 에 대한 테스트 데이터를 돌려 검증을 해보자.
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
# # argmax 는 해당  output에서 가장 index가 큰 결과를 가져온다. index가 크다는 의미는 가장 점수가 높게 설정되었다는
# # 말이고 해당 결과를 정답으로 볼 수 있다는 말이 된다. 예측한 값에서의 argmax와 실제 onehot encode에서의 argmax를
# # 각각 가져와서 비교해 보자. 해당 값이 같으면 true, 틀리면 false를 리턴할 것이다.
# # correct_prediction은 true, false 배열을 나타낸다.
# # correct_prediction에 대한 출력해 보고 싶다면 직접 호출할 수는 없고 session.run을 실행해서 결과를
# # 확인해야 한다. 아래와 같이 코드를 작성한 후 결과를 확인해 보자.
#
# print(sess.run(correct_prediction, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
#
# # accuracy는 위의 boolean 배열을 True일 경우에는 False일 경우에는 0으로 변환 한 후 평균을 구한 값이다.
# # 해당 결과를 확인해 보자.
#
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
# # mnist.test.images와 mnist.test.labels 의 실제 값들을 직접 보고 싶다면 아래와 같이 작성한 후 확인한다.
# # 실제로 1000번을 돌려서 해봤는데 jupyter가 뻗을 뻔 했다.... (print가 공수가 많이 드는 로직인가...)
# # range를 1로만 주고 확인해보자.
#
# for i in range(1):
#     batch_x, batch_y = mnist.test.next_batch(100)
#     diff_a = sess.run(tf.argmax(y,1), feed_dict={x:batch_x})
#     diff_b = sess.run(tf.argmax(y_,1), feed_dict={y_:batch_y})
#
#     print(diff_a)
#     print(diff_b)
#
# # 조금 더 보기 편하게 아래와 같이 수정하였다.
# for i in range(2):
#     result_boolean = []
#     batch_x, batch_y = mnist.test.next_batch(9)
#     diff_a = sess.run(tf.argmax(y,1), feed_dict={x:batch_x})
#     diff_b = sess.run(tf.argmax(y_,1), feed_dict={y_:batch_y})
#     print("sample output : " + str(diff_a))
#
#     for k in range(9):
#         if diff_a[k] == diff_b[k]:
#             result_boolean.append("T")
#         else:
#             result_boolean.append("F")
#     print("compare : " + str(result_boolean))
#
#     plt.figure(i)
#     coordi = [191,192,193,194,195,196,197,198,199]
#
#     for index, image in enumerate(batch_x):
#         image.shape(28,28)
#         plt.subplot(coordi[index])
#         plt.imshow(image)
# print("sample input :")