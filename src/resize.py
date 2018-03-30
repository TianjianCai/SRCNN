import tensorflow as tf
import matplotlib.pyplot as plt
import os

files = ['../img/' + str for str in sorted(os.listdir('../img/'))]
img_raw = tf.stack([tf.read_file(file) for file in files])


def decode_func(i):
    decoded = tf.cast(tf.image.decode_jpeg(i), tf.float32) / (2 ** 8)
    return decoded


img_decoded = tf.map_fn(fn=decode_func, elems=img_raw, dtype=tf.float32)


def resize_func(i):
    small_img = tf.image.resize_images(i, [tf.shape(img_decoded)[1] // 4, tf.shape(img_decoded)[2] // 4])
    large_img = tf.image.resize_images(small_img, [tf.shape(img_decoded)[1], tf.shape(img_decoded)[2]])
    return small_img


img_resized_X = tf.map_fn(fn=resize_func, elems=img_decoded, dtype=tf.float32)

img_size1 = tf.shape(img_resized_X)

large = tf.reshape(img_resized_X,[-1,32,1,32,1,3])

large = tf.tile(large,[1,1,4,1,4,1])

large = tf.reshape(large,[-1,128,128,3])

img_size2 = tf.shape(large)

linear = tf.lin_space(-10.0,10.0,100)

act = tf.nn.tanh(linear)

sess = tf.Session()

plt.plot(sess.run(act))
plt.show()

