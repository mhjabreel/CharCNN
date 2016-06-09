import tensorflow as tf
import numpy as np
from math import sqrt
import time

class CharConvNet(object):

    def __init__(self,
                conv_layers = [
                    [256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
                    ],
                 fully_layers = [1024, 1024],
                 l0 = 1014,
                 alphabet_size = 69,
                 no_of_classes = 4,
                 th = 1e-6):


    
        seed = time.time()
        
        tf.set_random_seed(seed)
        with tf.name_scope("Input-Layer"):
            #Model inputs
            self.input_x = tf.placeholder(tf.int64, shape = [None, l0], name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape = [None, no_of_classes], name = 'input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        with tf.name_scope("Embedding-Layer"), tf.device('/cpu:0'):
            #Quantization layer
            
            Q = tf.concat(0,
                          [
                              tf.zeros([1, alphabet_size]), # Zero padding vector for out of alphabet characters
                              tf.one_hot(range(alphabet_size), alphabet_size, 1.0, 0.0) # one-hot vector representation for alphabets
                           ],
                          name='Q')
            
            x = tf.nn.embedding_lookup(Q, self.input_x)
            x = tf.expand_dims(x, -1) # Add the channel dim, thus the shape of x is [batch_size, l0, alphabet_size, 1]
            
        var_id = 0
        

        
        # Convolution layers
        for i, cl in enumerate(conv_layers):
            var_id += 1 
            with tf.name_scope("ConvolutionLayer"):
                filter_width = x.get_shape()[2].value
                filter_shape = [cl[1], filter_width, 1, cl[0]] # Perform 1D conv with [kw, inputFrameSize (i.e alphabet_size), outputFrameSize]
                # Convolution layer
                stdv = 1/sqrt(cl[0]*cl[1])
                W = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv), dtype='float32', name='W' ) # The kernel of the conv layer is a trainable vraiable
                b = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name = 'b') # and the biases as well
                
                conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID", name='Conv') # Perform the convolution operation
                x = tf.nn.bias_add(conv, b)
                
                #Threshold
            with tf.name_scope("ThresholdLayer"):
                x = tf.select(tf.less(x, th), tf.zeros_like(x), x)

                

            if not cl[-1] is None:
                with tf.name_scope("MaxPoolingLayer" ):
                    # Maxpooling over the outputs
                    pool = tf.nn.max_pool(x, ksize=[1, cl[-1], 1, 1], strides=[1, cl[-1], 1, 1], padding='VALID')
                    x = tf.transpose(pool, [0, 1, 3, 2]) # [batch_size, img_width, img_height, 1]
            else:
                x = tf.transpose(x, [0, 1, 3, 2], name='tr%d' % var_id) # [batch_size, img_width, img_height, 1]
                

        with tf.name_scope("ReshapeLayer"):
            #Reshape layer
            vec_dim = x.get_shape()[1].value * x.get_shape()[2].value
            
            x = tf.reshape(x, [-1, vec_dim])

        
        
        weights = [vec_dim] + list(fully_layers) # The connection from reshape layer to fully connected layers
        
        for i, fl in enumerate(fully_layers):
            var_id += 1
            with tf.name_scope("LinearLayer" ):
                #Fully-Connected layer
                stdv = 1/sqrt(weights[i])
                W = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype='float32', name='W')
                b = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype='float32', name = 'b')
                
                x = tf.nn.xw_plus_b(x, W, b)
                
            with tf.name_scope("ThresholdLayer" ):
                x = tf.select(tf.less(x, th), tf.zeros_like(x), x)
                

            with tf.name_scope("DropoutLayer"):
                # Add dropout
                x = tf.nn.dropout(x, self.dropout_keep_prob)
                

        with tf.name_scope("OutputLayer"):
            stdv = 1/sqrt(weights[-1])
            #Output layer
            W = tf.Variable(tf.random_uniform([weights[-1], no_of_classes], minval=-stdv, maxval=stdv), dtype='float32', name='W')
            b = tf.Variable(tf.random_uniform(shape=[no_of_classes], minval=-stdv, maxval=stdv), name = 'b')
            
            self.p_y_given_x = tf.nn.xw_plus_b(x, W, b, name="scores")
            self.predictions = tf.argmax(self.p_y_given_x, 1)

        #with tf.name_scope("Loss"):


        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.p_y_given_x, self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("Accuracy"):
            # Accuracy
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
