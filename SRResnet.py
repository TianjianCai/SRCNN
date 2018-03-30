# import from system
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


class ResUnit(object):
    def __init__(self, x,
                 strides=1,
                 keep_prob=1,
                 training=True,
                 w1=None, w2=None, w3=None, b1=None, b2=None, b3=None,
                 k1=9, k2=1, k3=7, c1=3, c2=64, c3=32):
        if w1 is None:
            self.w1 = tf.Variable(tf.random_normal(shape=[k1, k1, c1, c2]))
            self.w2 = tf.Variable(tf.random_normal(shape=[k2, k2, c2, c3]))
            self.w3 = tf.Variable(tf.random_normal(shape=[k3, k3, c3, c1]))
            self.b1 = tf.Variable(tf.zeros([c2]))
            self.b2 = tf.Variable(tf.zeros([c3]))
            self.b3 = tf.Variable(tf.zeros([c1]))
        else:
            self.w1 = tf.Variable(w1)
            self.w2 = tf.Variable(w2)
            self.w3 = tf.Variable(w3)
            self.b1 = tf.Variable(b1)
            self.b2 = tf.Variable(b2)
            self.b3 = tf.Variable(b3)
        # bn1 = tf.contrib.layers.batch_norm(x, center=True, scale=False, is_training=training, updates_collections=None)
        bn1 = tf.map_fn(self.batch_norm, x)
        act1 = tf.nn.dropout(tf.nn.tanh(bn1), keep_prob)
        layer1 = self.conv2d(act1, self.w1, self.b1, strides)
        #bn2 = tf.contrib.layers.batch_norm(layer1, center=True, scale=False, is_training=training, updates_collections=None)
        bn2 = tf.map_fn(self.batch_norm, layer1)
        act2 = tf.nn.dropout(tf.nn.tanh(bn2), keep_prob)
        layer2 = self.conv2d(act2, self.w2, self.b2, strides)
        #bn3 = tf.contrib.layers.batch_norm(layer2, center=True, scale=False, is_training=training, updates_collections=None)
        bn3 = tf.map_fn(self.batch_norm, layer2)
        act3 = tf.nn.dropout(tf.nn.tanh(bn3), keep_prob)
        layer3 = self.conv2d(act3, self.w3, self.b3, strides)
        addition = tf.add(x, layer3)
        self.UnitOut = addition

    def conv2d(self, x, w, b, strides):
        y = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, b)
        return y

    def batch_norm(self, x):
        return tf.divide(tf.subtract(x, tf.multiply(tf.reduce_mean(x), 0.5)), tf.reduce_mean(x))


BIT_DEPTH = 8
RESIZE_K = 4
LEARNING_RATE = 1e-4
DROPOUT = 0.75
BATCH_SIZE = 4
K = 100
SHOW_PLT = False

files = ['img/' + str for str in sorted(os.listdir('img/'))]
file_length = np.shape(files)
file_length = file_length[0]

keep_prob = tf.placeholder(tf.float32)
batch_start = tf.placeholder(tf.int32)
batch_size = tf.placeholder(tf.int32)
is_training = tf.placeholder(tf.bool)

iteration_count = tf.Variable(0, dtype=tf.int64)
iteration_add = tf.assign(iteration_count,iteration_count+1000)

img_raw = tf.stack([tf.read_file(file) for file in files])

img_raw_batch = tf.slice(img_raw, [batch_start], [batch_size])


# img_decoded = [tf.image.rgb_to_hsv(tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH)) for i in img_raw]
def decode_func(i):
    decoded = tf.cast(tf.image.decode_jpeg(i), tf.float32) / (2 ** BIT_DEPTH)
    return decoded


img_decoded = tf.map_fn(fn=decode_func, elems=img_raw_batch, dtype=tf.float32)


def resize_func(i):
    small_img = tf.image.resize_images(i, [tf.shape(img_decoded)[1] // RESIZE_K, tf.shape(img_decoded)[2] // RESIZE_K])
    return small_img


img_resized_X = tf.map_fn(fn=resize_func, elems=img_decoded, dtype=tf.float32)
img_resized_X = tf.reshape(img_resized_X,[-1,tf.shape(img_resized_X)[1],1,tf.shape(img_resized_X)[2],1,3])
img_resized_X = tf.tile(img_resized_X,[1,1,RESIZE_K,1,RESIZE_K,1])
img_resized_X = tf.reshape(img_resized_X,[-1,tf.shape(img_decoded)[1],tf.shape(img_decoded)[2],3])

res1 = ResUnit(x=img_resized_X)
res2 = ResUnit(x=res1.UnitOut)
res3 = ResUnit(x=res2.UnitOut)

cost = tf.reduce_mean(tf.square(res3.UnitOut - img_decoded))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=cost)

init = tf.global_variables_initializer()

config = tf.ConfigProto(
    device_count={'GPU': 1}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

saver = tf.train.Saver()

try:
    saver.restore(sess,'C:\\Users\\CTJ\\Documents\\GitHub\\SRCNN\\log\\resnet.ckpt')
    print('checkpoint loaded')
except:
    print('cannot load checkpoint')

print(sess.run(tf.reduce_mean(tf.square(img_resized_X - img_decoded)), {batch_start: 1, batch_size: 1}))

if SHOW_PLT is True:
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    p1 = fig.add_subplot(1, 3, 1)
    p2 = fig.add_subplot(1, 3, 2)
    p3 = fig.add_subplot(1, 3, 3)
    p1.imshow(sess.run(img_decoded, {batch_start: 1, batch_size: 1})[0], interpolation='none')
    p3.imshow(sess.run(img_resized_X, {batch_start: 1, batch_size: 1})[0], interpolation='none')

i = 0
file_start = 0
file_size = BATCH_SIZE
while True:

    if i % 1000 == 0:
        saver.save(sess, 'C:\\Users\\CTJ\\Documents\\GitHub\\SRCNN\\log\\resnet.ckpt')
        c = sess.run(cost, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False})
        print('iteration ' + repr(sess.run(iteration_count)) + ', cost: ' + repr(c))
        i2 = sess.run(res3.UnitOut, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False})
        with open('mse.csv','a+') as f:
            content = repr(sess.run(iteration_count)) + ',' + repr(c) + '\n'
            f.write(content)
            f.close()
        '''
        print(sess.run(l1.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
        print(sess.run(l2.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
        print(sess.run(l3.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
        '''
        sess.run(iteration_add)
        if SHOW_PLT is True:
            p2.imshow(np.clip(i2[0],0,1), interpolation='none')
            fig.canvas.draw()
            plt.pause(0.1)
    i = i + 1
    sess.run(train_op, {keep_prob: DROPOUT, batch_start: file_start, batch_size: file_size, is_training: True})
    file_start = file_start + BATCH_SIZE
    if file_start + file_size > file_length - 1:
        file_start = 0
