# import from system
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#import from src
sys.path.append('src')
import Layers
import psnr


BIT_DEPTH = 8
RESIZE_K = 2
LEARNING_RATE = 0.001

files = ['img/'+str for str in sorted(os.listdir('img/'))]

img_raw = [tf.read_file(file) for file in files]
img_decoded = [tf.cast(tf.image.decode_png(i),tf.float32)/(2**BIT_DEPTH) for i in img_raw]

img_resized_X = [
    tf.image.resize_images(
        tf.image.resize_images(i,[tf.shape(img_decoded[0])[0]//RESIZE_K,tf.shape(img_decoded[0])[1]//RESIZE_K]),
        [tf.shape(img_decoded[0])[0],tf.shape(img_decoded[0])[1]]
    )
    for i in img_decoded]

l1 = Layers.patch_extraction(x=img_resized_X,channels=4)
l2 = Layers.none_linear_mapping(x=l1.out,channels=4)
l3 = Layers.reconstruction(x=l2.out,channels=4)

cost = tf.reduce_mean(tf.squared_difference(l3.out, img_decoded))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=cost,var_list=[l1.W,l1.B,l2.W,l2.B,l3.W,l3.B])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

i = 0
while True:
    if i%100 == 0:
        print(sess.run(cost))
        i1 = np.array(sess.run(img_decoded))
        i2 = np.array(sess.run(l3.out))
        #print(i2)
        fig=plt.figure()
        fig.add_subplot(2,1,1)
        plt.imshow(np.reshape(i1[0,:,:,0:3],[92,100,3]))
        fig.add_subplot(2,1,2)
        plt.imshow(np.reshape(i2[0,:,:,0:3],[92,100,3]))
        plt.pause(0.5)
        plt.close()
    i = i+1
    sess.run(train_op)




