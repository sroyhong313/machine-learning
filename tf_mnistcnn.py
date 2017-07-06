from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#begin interactive session
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 784])
y_= tf.placeholder(tf.float32, shape = [None, 10])

# Weight W and bias b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10])) # b is 10-dim (10 x 1)

# Initialize variables
sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,
                                                                       logits = y))

# Gradient descent with learning rate = 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x : batch[0], y_: batch[1]})

# evaluate model -- how many did it get it right?
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#evaluate accuracy on test data
print (accuracy.eval(feed_dict = {x : mnist.test.images, y_ : mnist.test.labels}))

# randomize weights and bias for symmetry breaking
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
