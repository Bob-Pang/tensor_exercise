#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pb
# @File    : helloworld.py

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

Weight = tf.Variable(tf.random_uniform([784,10],minval=-1,maxval=1,dtype=tf.float32,name='Weight'))
bias = tf.Variable(tf.ones([10]),dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='input_sample')
# sig = tf.matmul(x,Weight) + bias
# #softmax output
# y = tf.exp(sig) / tf.reduce_sum(sig,reduction_indices=[1])
y = tf.nn.softmax(tf.matmul(x,Weight) + bias)
true_label = tf.placeholder(dtype=tf.float32,shape=[None,10],name='true_label')

#定义损失函数,使用的损失函数为交叉商损失函数
#tf.multiply 是向量的点乘, 点乘的乘法中两个张量的类型必须相同
cross_encropy_loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(true_label,tf.log(y)),reduction_indices=[1]))

#设置优化函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_encropy_loss)

#建立tensorflow启动时客户端和服务端连接需要的session
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

#训练1000次
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,true_label:batch_ys})

#进行模型的评估
correct_prediction = tf.equal(tf.argmax(true_label,axis=1),tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

acc = sess.run(accuracy,feed_dict={x:mnist.test.images,true_label:mnist.test.labels})
print(acc)


