import tensorflow as tf

class patch_extraction(object):
    def __init__(self,x,kernel_size=5,strides=1,channels=3,W=None,B=None):
        if W is None:
            self.W = tf.Variable(tf.random_normal([kernel_size,kernel_size,channels,channels]))
        else:
            self.W = W
        if B is None:
            self.B = tf.Variable(tf.zeros([channels]))
        else:
            self.B = B
        y = tf.nn.conv2d(x, self.W, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, self.B)
        self.out = tf.nn.relu(y)