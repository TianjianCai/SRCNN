# import from system
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import os
import sys

#import from src
sys.path.append('src')
import Layers
import psnr


BIT_DEPTH = 8
RESIZE_K = 4
LEARNING_RATE = 0.0001
DROPOUT = 0.75

files = ['img/'+str for str in sorted(os.listdir('img/'))]

keep_prob = tf.placeholder(tf.float32)

img_raw = [tf.read_file(file) for file in files]
#img_decoded = [tf.image.rgb_to_hsv(tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH)) for i in img_raw]
img_decoded = [tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH) for i in img_raw]

img_resized_X = [
    tf.image.resize_images(
        tf.image.resize_images(i,[tf.shape(img_decoded[0])[0]//RESIZE_K,tf.shape(img_decoded[0])[1]//RESIZE_K]),
        [tf.shape(img_decoded[0])[0],tf.shape(img_decoded[0])[1]]
    )
    for i in img_decoded]

l1 = Layers.patch_extraction(x=img_resized_X,channels_in=3,channels_out=64,keep_prob=keep_prob)
l2 = Layers.none_linear_mapping(x=l1.out,channels_in=64,channels_out=32,keep_prob=keep_prob)
l3 = Layers.reconstruction(x=l2.out,channels_in=32,channels_out=3)

cost = tf.reduce_mean(tf.squared_difference(l3.out, img_decoded))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=cost,var_list=[l1.W,l1.B,l2.W,l2.B,l3.W,l3.B])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(tf.reduce_mean(tf.squared_difference(img_resized_X, img_decoded))))

plt.ion()
fig=plt.figure(figsize=(15,5))
p1 = fig.add_subplot(1,3,1)
p2 = fig.add_subplot(1,3,2)
p3 = fig.add_subplot(1,3,3)
p1.imshow(sess.run(img_decoded)[0],interpolation='none')
p3.imshow(sess.run(img_resized_X)[0],interpolation='none')
i = 0
while True:
    if i%1000 == 0:
        if i%10000 == 0:
            np.save('weights', [sess.run(l1.W),sess.run(l1.B),sess.run(l2.W),sess.run(l2.B),sess.run(l3.W),sess.run(l3.B)])
        c = sess.run(cost,{keep_prob:1})
        if c < 0.0001:
            break
        print('epoch ' + repr(i) + ', cost: ' + repr(c))
        i2 = sess.run(l3.out,{keep_prob:1})
        '''
        p1.imshow(hsv_to_rgb(i1[0]),interpolation='none')
        p2.imshow(hsv_to_rgb(i2[0]),interpolation='none')
        p3.imshow(hsv_to_rgb(i3[0]),interpolation='none')
        '''
        #p1.imshow(i1[0],interpolation='none')
        p2.imshow(i2[0],interpolation='none')
        #p3.imshow(i3[0],interpolation='none')
        fig.canvas.draw()
        plt.pause(0.00001)
    i = i+1
    sess.run(train_op,{keep_prob:DROPOUT})




