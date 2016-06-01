import tensorflow as tf
import numpy as np
from math import sqrt

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
                 full_layers = [1024, 1024],
                 l0 = 1014,
                 alphabet_size = 69,
                 no_of_classes = 4,
                 th = 1e-6):


        
        self.dropout_prop = tf.placeholder(tf.float32, name="p")
        
        with tf.name_scope("InputLayer"):
            self.input_x = tf.placeholder(tf.int64, shape = [None, l0], name = 'input_x')
            self.input_y = tf.placeholder(tf.float32, shape = [None, no_of_classes], name = 'input_y')


        with tf.name_scope('QuantizationLayer'), tf.device('/cpu:0'):
            Q = tf.concat(0,
                          [
                              tf.zeros([1, alphabet_size]), # Zero padding vector for out of alphabet characters
                              tf.one_hot(range(alphabet_size), alphabet_size, 1.0, 0.0) # one-hot vector representation for alphabets
                           ],
                          name='Q')

            x = tf.nn.embedding_lookup(Q, self.input_x)
            x = tf.expand_dims(x, -1) # Add the channel dim, thus the shape of x is [batch_size, l0, alphabet_size, 1]


        for i, cl in enumerate(conv_layers):
            with tf.name_scope("ConvLayer%d" %(i + 1)):
                filter_width = x.get_shape()[2].value
                filter_shape = [cl[1], filter_width, 1, cl[0]] # Perform 1D conv with [kw, inputFrameSize (i.e alphabet_size), outputFrameSize]

                # Initiate the weights and biases
                stdv = 1 / sqrt(cl[0] * cl[1]) # cl[0] = outputFrameSize, cl[1] = kw
                W = tf.Variable(tf.random_uniform(filter_shape, minval = -stdv, maxval = stdv), name = 'W') # The kernel of the conv layer is a trainable vraiable
                b = tf.Variable(tf.random_uniform([cl[0]], minval = -stdv, maxval = stdv), name = 'b') # and the biases as well

                x = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID") # Perform the convolution operation
                x = tf.nn.bias_add(x, b)

            #Add threshold layer
            with tf.name_scope("Conv-ThresholdLayer%d" % (i + 1)):
                x = tf.select(tf.less(x, th), tf.zeros_like(x), x, name = 'Threshold')

                
            if not cl[-1] is None:
                # Add MaxPooling Layer
                with tf.name_scope("MaxPoolingLayer%d" % (i + 1)):
                    x = tf.nn.max_pool(x, ksize = [1, cl[-1], 1, 1], strides = [1, cl[-1], 1, 1], padding='VALID')

            x = tf.transpose(x, [0, 1, 3, 2]) # [batch_size, width, height, 1]


        #Add reshape layer

        with tf.name_scope("ReshapeLayer"):
            vec_dim = x.get_shape()[1].value * x.get_shape()[2].value
            x = tf.reshape(x, [-1, vec_dim])


        weights = [vec_dim] + list(full_layers) # The connection from reshape layer to fully connected layers

        for i, fl in enumerate(full_layers):

            stdv = 1 / sqrt(weights[i])
            # Add linear layer
            with tf.name_scope("LinearLayer%d" % (i + 1)):
                W = tf.Variable(tf.random_uniform([weights[i], fl], minval = -stdv, maxval = stdv), name = 'W')
                b = tf.Variable(tf.random_uniform([fl], minval = -stdv, maxval = stdv), name = 'b')
            
                x = tf.nn.xw_plus_b(x, W, b)
            # Add threshold layer
            with tf.name_scope("Linear-ThresholdLayer%d" % (i + 1)):
                x = tf.select(tf.less(x, th), tf.zeros_like(x), x, name = 'Threshold')

            # Add dropout layer
            with tf.name_scope("DropoutLayer%d" % (i + 1)):
                x = tf.nn.dropout(x, self.dropout_prop)


        # The output layer

        with tf.name_scope("OutputLayer"):
            stdv = 1 / sqrt(weights[-1])
            W = tf.Variable(tf.random_uniform([weights[-1], no_of_classes], minval = -stdv, maxval = stdv), name='W')
            b = tf.Variable(tf.random_uniform([no_of_classes], minval = -stdv, maxval = stdv), name = 'b')

            p_y_given_x = tf.nn.xw_plus_b(x, W, b)
            predictions = tf.argmax(p_y_given_x, 1)


        with tf.name_scope("Loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(p_y_given_x, self.input_y)
            self.loss = tf.reduce_mean(losses, name = 'loss')

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = 'accuracy')

