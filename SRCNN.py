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


BIT_DEEPTH = 8
files_X = ['img/X/'+str for str in sorted(os.listdir('img/X/'))]
files_Y = ['img/Y/'+str for str in sorted(os.listdir('img/Y/'))]

img_raw_X = [tf.read_file(file) for file in files_X]
img_raw_Y = [tf.read_file(file) for file in files_Y]
img_decoded_X = [tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEEPTH) for i in img_raw_X]
img_decoded_Y = [tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEEPTH) for i in img_raw_Y]
img_resized_X = [tf.image.resize_images(i,[tf.shape(tf.image.decode_jpeg(img_raw_Y[0]))[0],tf.shape(tf.image.decode_jpeg(img_raw_Y[0]))[1]]) for i in img_decoded_X]

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

i1 = sess.run(img_decoded_X)
i2 = sess.run(img_resized_X)
i3 = sess.run(img_decoded_Y)
print(np.shape(i1))
print(np.shape(i2))
print(np.shape(i3))
print(psnr.psnr(i2[0], i3[0]))
print(psnr.psnr(i3[0], i3[0]))
fig=plt.figure()
fig.add_subplot(3,1,1)
plt.imshow(i1[0])
fig.add_subplot(3,1,2)
plt.imshow(i2[0])
fig.add_subplot(3,1,3)
plt.imshow(i3[0])
plt.show()

