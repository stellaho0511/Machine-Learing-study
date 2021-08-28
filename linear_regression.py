import tensorflow as tf
W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

@tf.function
def linear_model(x):
  return W*x + b

# MSE 손실함수 \mean{(y' - y)^2}
@tf.function
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))

# gradient descent 옵티마이저 정의 / a = 0.01
optimizer = tf.optimizers.SGD(0.01)

# 최적화를 위한 함수 정의
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = linear_model(x)
    loss = mse_loss(y_pred, y)
  gradients = tape.gradient(loss, [W, b])
  optimizer.apply_gradients(zip(gradients, [W, b]))

# 트레이닝을 위한 데이터
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# gradient descent 수행
for i in range(1000):
  train_step(x_train, y_train)

# 테스트를 위한 입력값
x_test = [3.5, 5, 5.5, 6]
print(linear_model(x_test).numpy())
