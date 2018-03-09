import tensorflow as tf

class patch_extraction(object):
    def __init__(self,x,kernel_size=9,strides=1,channels=3,W=None,B=None):
        if W is None:
            #self.W = tf.Variable(tf.random_uniform(shape=[kernel_size,kernel_size,channels,channels],minval=0,maxval=1))
            self.W = tf.Variable(tf.random_normal(shape=[kernel_size,kernel_size,channels,channels]))
        else:
            self.W = W
        if B is None:
            self.B = tf.Variable(tf.zeros([channels]))
        else:
            self.B = B
        y = tf.nn.conv2d(x, self.W, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, self.B)
        self.out = tf.nn.relu(y)
        
class none_linear_mapping(object):
    def __init__(self,x,kernel_size=1,strides=1,channels=3,W=None,B=None):
        if W is None:
            #self.W = tf.Variable(tf.random_uniform(shape=[kernel_size,kernel_size,channels,channels],minval=0,maxval=1))
            self.W = tf.Variable(tf.random_normal(shape=[kernel_size,kernel_size,channels,channels]))
        else:
            self.W = W
        if B is None:
            self.B = tf.Variable(tf.zeros([channels]))
        else:
            self.B = B
        y = tf.nn.conv2d(x, self.W, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, self.B)
        self.out = tf.nn.relu(y)

class reconstruction(object):
    def __init__(self,x,kernel_size=5,strides=1,channels=3,W=None,B=None):
        if W is None:
            #self.W = tf.Variable(tf.random_uniform(shape=[kernel_size,kernel_size,channels,channels],minval=0,maxval=1))
            self.W = tf.Variable(tf.random_normal(shape=[kernel_size,kernel_size,channels,channels]))
        else:
            self.W = W
        if B is None:
            self.B = tf.Variable(tf.zeros([channels]))
        else:
            self.B = B
        y = tf.nn.conv2d(x, self.W, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, self.B)
        '''
        ls0 = tf.less(y,0)
        gt1 = tf.greater(y,1)
        y = tf.where(ls0,tf.zeros_like(y),y)
        y = tf.where(gt1,tf.ones_like(y),y)
        '''
        self.out = tf.nn.sigmoid(y)