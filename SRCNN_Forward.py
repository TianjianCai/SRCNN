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
BATCH_SIZE = 5

files = ['img_test/'+str for str in sorted(os.listdir('img_test/'))]
file_length = np.shape(files)
file_length = file_length[0]

keep_prob = tf.placeholder(tf.float32)
batch_start = tf.placeholder(tf.int32)
batch_size = tf.placeholder(tf.int32)

img_raw = tf.stack([tf.read_file(file) for file in files])

img_raw_batch = tf.slice(img_raw,[batch_start],[batch_size])

#img_decoded = [tf.image.rgb_to_hsv(tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH)) for i in img_raw]
def decode_func(i):
    decoded = tf.cast(tf.image.decode_jpeg(i),tf.float32)/(2**BIT_DEPTH)
    return decoded
img_decoded = tf.map_fn(fn=decode_func,elems=img_raw_batch,dtype=tf.float32)

def resize_func(i):
    small_img = tf.image.resize_images(i,[tf.shape(img_decoded)[1]//RESIZE_K,tf.shape(img_decoded)[2]//RESIZE_K])
    large_img = tf.image.resize_images(small_img,[tf.shape(img_decoded)[1],tf.shape(img_decoded)[2]])
    return large_img
img_resized_X = tf.map_fn(fn=resize_func, elems=img_decoded,dtype=tf.float32)

W1,B1,W2,B2,W3,B3 = np.load('weights.npy')

l1 = Layers.patch_extraction(x=img_resized_X,channels_in=3,channels_out=64,keep_prob=keep_prob,W=W1,B=B1)
l2 = Layers.none_linear_mapping(x=l1.out,channels_in=64,channels_out=32,keep_prob=keep_prob,W=W2,B=B2)
l3 = Layers.reconstruction(x=l2.out,channels_in=32,channels_out=3,W=W3,B=B3)


init = tf.global_variables_initializer()

config = tf.ConfigProto(
    device_count = {'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(init)


pic_num = 0
fig=plt.figure(figsize=(15,5))
p1 = fig.add_subplot(1,3,1)
p2 = fig.add_subplot(1,3,2)
p3 = fig.add_subplot(1,3,3)
p1.imshow(sess.run(img_decoded,{batch_start:pic_num,batch_size:1})[0],interpolation='none')
p3.imshow(sess.run(img_resized_X,{batch_start:pic_num,batch_size:1})[0],interpolation='none')
i2 = sess.run(l3.out,{keep_prob:1,batch_start:pic_num,batch_size:1})
p2.imshow(i2[0],interpolation='none')
plt.show()




