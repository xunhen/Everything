import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import time
from matplotlib import pyplot as plt


def main():
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    grapg = tf.Graph()
    with grapg.as_default():
        with tf.variable_scope('placeholder'):
            image_place = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='images')
        net = image_place
        with slim.arg_scope([slim.convolution1d, slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer,
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(net, 2, slim.conv2d, 16, 5, scope='conv1')
            net = slim.pool(net, 2, stride=2, pooling_type='AVG', scope='pool1')
            net = slim.repeat(net, 3, slim.conv2d, 32, 7, scope='conv2')
            net = slim.pool(net, 2, stride=2, pooling_type='AVG', scope='pool2')

        init = tf.global_variables_initializer()

    sess = tf.Session(graph=grapg)
    sess.run(init)
    batch_size = np.random.randint(1, 10, size=[10, ])
    batch_size = [11, 8, 9, 10, 8, 9, 10, 3, 4, 3]
    batch_size = np.arange(0, 100) + 1
    batch_size = np.tile(batch_size.reshape([-1, 1]), [1, 2]).reshape([-1, ])
    print(batch_size)
    images = []
    for i in batch_size:
        images.append(np.random.rand(i, 124, 124, 3))

    flag = True
    for image in images:
        if flag:
            x1.append(image.shape[0])
        else:
            x2.append(image.shape[0])
        print(image.shape)
        tic = time.time()
        sess.run(net, feed_dict={image_place: image})
        toc = time.time()
        time_consumer = (toc - tic) * 1000
        if flag:
            y1.append(time_consumer)
        else:
            y2.append(time_consumer)
        flag = not flag
        print(time_consumer)
    plt.xlabel('batch_size')
    plt.ylabel('time')
    plt.plot(x1, y1, label='first')
    plt.plot(x2, y2, label='second')
    plt.draw()
    plt.show()
    pass


if __name__ == '__main__':
    main()
