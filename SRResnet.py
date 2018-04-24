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
                 k1=9, k2=3, k3=5, c1=3, c2=64, c3=32):
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
        self.addition = addition
        all_zeros = tf.zeros(tf.shape(addition))
        weight_cost_0 = 0 - tf.clip_by_value(tf.reduce_sum(tf.where(tf.greater(all_zeros, addition), addition, all_zeros)), -1e10, 0)
        weight_cost_1 = tf.clip_by_value(tf.reduce_sum(tf.where(tf.less(all_zeros, addition - 1), addition - 1, all_zeros)), 0, 1e10)
        self.weight_cost = weight_cost_0 + weight_cost_1
        self.UnitOut = addition

    def conv2d(self, x, w, b, strides):
        y = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, b)
        return y

    def batch_norm(self, x):
        return tf.divide(tf.subtract(x,tf.reduce_mean(x)), tf.clip_by_value(tf.reduce_max(x)-tf.reduce_min(x), 1e-10, 1e10))


BIT_DEPTH = 8
RESIZE_K = 2
LEARNING_RATE = 1e-6
DROPOUT = 0.75
BATCH_SIZE = 6
K = 100
SHOW_PLT = False
RESNUM = 3
ITERATION_NUM = 1000

files = ['img/' + str for str in sorted(os.listdir('img/'))]
file_length = np.shape(files)
file_length = file_length[0]

keep_prob = tf.placeholder(tf.float32)
batch_start = tf.placeholder(tf.int32)
batch_size = tf.placeholder(tf.int32)
is_training = tf.placeholder(tf.bool)
shuffle = tf.placeholder(tf.bool)

iteration_count = tf.Variable(0, dtype=tf.int64)
iteration_add = tf.assign(iteration_count,iteration_count+ITERATION_NUM)

img_raw = tf.stack([tf.read_file(file) for file in files])
img_raw_not_shuffled = img_raw

shuffle_op = tf.random_shuffle(img_raw)

img_raw_batch = tf.slice(tf.cond(shuffle,lambda: img_raw, lambda: img_raw_not_shuffled), [batch_start], [batch_size])


# img_decoded = [tf.image.rgb_to_hsv(tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH)) for i in img_raw]
def decode_func(i):
    decoded = tf.cast(tf.image.decode_jpeg(i), tf.float32) / (2 ** BIT_DEPTH)
    return decoded


img_decoded = tf.map_fn(fn=decode_func, elems=img_raw_batch, dtype=tf.float32)


def resize_func(i):
    small_img = tf.image.resize_images(i, [tf.shape(img_decoded)[1] // RESIZE_K, tf.shape(img_decoded)[2] // RESIZE_K])
    return small_img


def resize_func2(i):
    small_img = tf.image.resize_images(i, [tf.shape(img_decoded)[1] // RESIZE_K, tf.shape(img_decoded)[2] // RESIZE_K])
    large_img = tf.image.resize_images(small_img, [tf.shape(img_decoded)[1], tf.shape(img_decoded)[2]])
    return large_img


img_resized_X2 = tf.map_fn(fn=resize_func2, elems=img_decoded, dtype=tf.float32)

img_resized_X = tf.map_fn(fn=resize_func, elems=img_decoded, dtype=tf.float32)
img_resized_X = tf.reshape(img_resized_X,[-1,tf.shape(img_resized_X)[1],1,tf.shape(img_resized_X)[2],1,3])
img_resized_X = tf.tile(img_resized_X,[1,1,RESIZE_K,1,RESIZE_K,1])
img_resized_X = tf.reshape(img_resized_X,[-1,tf.shape(img_decoded)[1],tf.shape(img_decoded)[2],3])


reslist = []
reslist.append(ResUnit(x=img_resized_X2))
cost = reslist[0].weight_cost
for i in range(1, RESNUM):
    reslist.append(ResUnit(x=reslist[i-1].UnitOut))
    cost = tf.add(cost, reslist[i].weight_cost)
resout = reslist[len(reslist)-1]

cost_entropy = tf.reduce_mean(tf.clip_by_value(tf.subtract(0., tf.log(tf.clip_by_value(tf.subtract(1., tf.abs(tf.subtract(img_decoded, resout.UnitOut))), 1e-5, 1.))), 0., 1e5))

cost_old = cost + tf.reduce_mean(tf.square(resout.UnitOut - img_decoded))
cost = cost + cost_entropy

input_mse = tf.reduce_mean(tf.square(img_resized_X - img_decoded))
default_mse = tf.reduce_mean(tf.square(img_resized_X2 - img_decoded))

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

print(sess.run([input_mse, default_mse], {batch_start: 5, batch_size: 1, shuffle: False}))

if SHOW_PLT is True:
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    p1 = fig.add_subplot(1, 3, 1)
    p2 = fig.add_subplot(1, 3, 2)
    p3 = fig.add_subplot(1, 3, 3)
    p1.imshow(sess.run(img_decoded, {batch_start: 5, batch_size: 1, shuffle: False})[0], interpolation='none')
    p3.imshow(sess.run(img_resized_X, {batch_start: 5, batch_size: 1, shuffle: False})[0], interpolation='none')

i = 0
file_start = 0
file_size = BATCH_SIZE
while True:

    if i % ITERATION_NUM == 0:
        sess.run(shuffle_op, {shuffle: False})
        saver.save(sess, 'C:\\Users\\CTJ\\Documents\\GitHub\\SRCNN\\log\\resnet.ckpt')
        c = sess.run(cost, {keep_prob: 1, batch_start: 5, batch_size: 1, is_training: False, shuffle: False})
        c2 = sess.run(resout.weight_cost, {keep_prob: 1, batch_start: 5, batch_size: 1, is_training: False, shuffle: False})
        ce = sess.run(cost_entropy, {keep_prob: 1, batch_start: 5, batch_size: 1, is_training: False, shuffle: False})
        mse = sess.run(cost_old, {keep_prob: 1, batch_start: 5, batch_size: 1, is_training: False, shuffle: False})
        print('iteration ' + repr(sess.run(iteration_count)) + ', cost: ' + repr(mse) + ', weight cost: ' + repr(c2) + ', entropy: ' + repr(ce))
        i2 = sess.run(resout.UnitOut, {keep_prob: 1, batch_start: 5, batch_size: 1, is_training: False, shuffle:False})
        #print(sess.run(resout.addition[0], {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
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
    sess.run(shuffle_op, {shuffle: True})
    sess.run(train_op, {keep_prob: DROPOUT, batch_start: file_start, batch_size: file_size, shuffle: True})
    file_start = file_start + BATCH_SIZE
    if file_start + file_size > file_length - 1:
        file_start = 0
