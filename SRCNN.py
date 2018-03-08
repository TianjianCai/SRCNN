import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('src')
import Layers
import time

file_queue = tf.train.string_input_producer(['img/'+str for str in os.listdir('img')])
reader = tf.WholeFileReader()

_,img_raw = reader.read(file_queue)
img = tf.cast(tf.image.decode_jpeg(img_raw)/255,tf.float32)

#l1 = Layers.patch_extraction([img])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

threads = tf.train.start_queue_runners(sess=sess)
#img_out = sess.run(l1.out)
    
#print(np.shape(img_in))
for i in range(10):
    img_in = sess.run(img)
    print(np.shape(img_in))
    time.sleep(1)
