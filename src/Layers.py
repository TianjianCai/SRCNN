import tensorflow as tf

class patch_extraction(object):
    def __init__(self,x,kernel_size=9,strides=1,channels_in=3,channels_out=3,W=None,B=None,keep_prob=1,is_training=True):
        if W is None:
            #self.W = tf.Variable(tf.random_uniform(shape=[kernel_size,kernel_size,channels,channels],minval=0,maxval=1))
            self.W = tf.Variable(tf.random_normal(shape=[kernel_size,kernel_size,channels_in,channels_out]))
        else:
            self.W = tf.Variable(tf.convert_to_tensor(W))
        if B is None:
            self.B = tf.Variable(tf.zeros([channels_out]))
        else:
            self.B = tf.Variable(tf.convert_to_tensor(B))
        y = tf.nn.conv2d(x, self.W, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, self.B)
        #y = tf.contrib.layers.batch_norm(y,center=False,scale=True,is_training=is_training,updates_collections=None)
        y = tf.nn.sigmoid(y)
        self.out = tf.nn.dropout(y,keep_prob)
        
class none_linear_mapping(object):
    def __init__(self,x,kernel_size=1,strides=1,channels_in=3,channels_out=3,W=None,B=None,keep_prob=1,is_training=True):
        if W is None:
            #self.W = tf.Variable(tf.random_uniform(shape=[kernel_size,kernel_size,channels,channels],minval=0,maxval=1))
            self.W = tf.Variable(tf.random_normal(shape=[kernel_size,kernel_size,channels_in,channels_out]))
        else:
            self.W = W
        if B is None:
            self.B = tf.Variable(tf.zeros([channels_out]))
        else:
            self.B = B
        y = tf.nn.conv2d(x, self.W, strides=[1, strides, strides, 1], padding='SAME')
        y = tf.nn.bias_add(y, self.B)
        #y = tf.contrib.layers.batch_norm(y,center=True,scale=True,is_training=is_training,updates_collections=None)
        y = tf.nn.sigmoid(y)
        self.out = tf.nn.dropout(y,keep_prob)

class reconstruction(object):
    def __init__(self,x,kernel_size=7,strides=1,channels_in=3,channels_out=3,W=None,B=None,keep_prob=1,is_training=True):
        if W is None:
            #self.W = tf.Variable(tf.random_uniform(shape=[kernel_size,kernel_size,channels,channels],minval=0,maxval=1))
            self.W = tf.Variable(tf.random_normal(shape=[kernel_size,kernel_size,channels_in,channels_out]))
        else:
            self.W = W
        if B is None:
            self.B = tf.Variable(tf.zeros([channels_out]))
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
        #y = tf.contrib.layers.batch_norm(y,center=False,scale=True,is_training=is_training,updates_collections=None)
        y = tf.nn.sigmoid(y)
        self.out = y