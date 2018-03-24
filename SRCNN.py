# import from system
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# import from src
sys.path.append('src')
import Layers

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

img_raw = tf.stack([tf.read_file(file) for file in files])

img_raw_batch = tf.slice(img_raw, [batch_start], [batch_size])


# img_decoded = [tf.image.rgb_to_hsv(tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH)) for i in img_raw]
def decode_func(i):
    decoded = tf.cast(tf.image.decode_jpeg(i), tf.float32) / (2 ** BIT_DEPTH)
    return decoded


img_decoded = tf.map_fn(fn=decode_func, elems=img_raw_batch, dtype=tf.float32)


def resize_func(i):
    small_img = tf.image.resize_images(i, [tf.shape(img_decoded)[1] // RESIZE_K, tf.shape(img_decoded)[2] // RESIZE_K])
    large_img = tf.image.resize_images(small_img, [tf.shape(img_decoded)[1], tf.shape(img_decoded)[2]])
    return large_img


img_resized_X = tf.map_fn(fn=resize_func, elems=img_decoded, dtype=tf.float32)

'''
Loading weight file
'''
try:
    W1, B1, W2, B2, W3, B3 = np.load('weights.npy')
    W1 = tf.Variable(W1)
    B1 = tf.Variable(B1)
    W2 = tf.Variable(W2)
    B2 = tf.Variable(B2)
    W3 = tf.Variable(W3)
    B3 = tf.Variable(B3)
    l1 = Layers.patch_extraction(x=img_resized_X, channels_in=3, channels_out=64, keep_prob=keep_prob, W=W1, B=B1,
                                 is_training=is_training)
    l2 = Layers.none_linear_mapping(x=l1.out, channels_in=64, channels_out=32, keep_prob=keep_prob, W=W2, B=B2,
                                    is_training=is_training)
    l3 = Layers.reconstruction(x=l2.out, channels_in=32, channels_out=3, W=W3, B=B3, is_training=is_training)
    print('weight loaded successfully')
except:
    l1 = Layers.patch_extraction(x=img_resized_X, channels_in=3, channels_out=64, keep_prob=keep_prob,
                                 is_training=is_training)
    l2 = Layers.none_linear_mapping(x=l1.out, channels_in=64, channels_out=32, keep_prob=keep_prob,
                                    is_training=is_training)
    l3 = Layers.reconstruction(x=l2.out, channels_in=32, channels_out=3, is_training=is_training)
    print('No weight file found,use random weight')

cost = tf.reduce_mean(tf.square(l3.out - img_decoded))
cost_weight = cost + K*l1.weight_cost + K*l2.weight_cost + K*l3.weight_cost

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=cost, var_list=[l1.W, l1.B, l2.W, l2.B, l3.W, l3.B])

init = tf.global_variables_initializer()

config = tf.ConfigProto(
    device_count={'GPU': 1}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

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
        # print(sess.run(cost_ce,{keep_prob:DROPOUT,batch_start:file_start,batch_size:file_size}))
        np.save('weights',
                [sess.run(l1.W), sess.run(l1.B), sess.run(l2.W), sess.run(l2.B), sess.run(l3.W), sess.run(l3.B)])
        c = sess.run(cost, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False})
        if c < 0.0001:
            pass
            # break
        print('iteration ' + repr(i) + ', cost: ' + repr(c))
        with open('mse.csv','a+') as f:
            content = repr(i) + ',' + repr(c) + '\n'
            f.write(content)
            f.close()
        i2 = sess.run(l3.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False})
        '''
        print(sess.run(l1.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
        print(sess.run(l2.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
        print(sess.run(l3.out, {keep_prob: 1, batch_start: 1, batch_size: 1, is_training: False}))
        '''
        if SHOW_PLT is True:
            p2.imshow(i2[0], interpolation='none')
            fig.canvas.draw()
            plt.pause(0.1)
    i = i + 1
    sess.run(train_op, {keep_prob: DROPOUT, batch_start: file_start, batch_size: file_size, is_training: True})
    file_start = file_start + BATCH_SIZE
    if file_start + file_size > file_length - 1:
        file_start = 0
