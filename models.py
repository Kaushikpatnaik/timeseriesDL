'''
Using the resnet implementation by Google to be an inspiration for the code
'''

import tensorflow as tf
import numpy as np
import pandas as pd


class oneDCNN(object):

    def __init__(self,args):
        '''
        Gather all the configurations required to make the model

        number of layers
        conv sizes per layer
        conv options - stride
        conv type
        batch size
        number of input channels

        '''

        self.hidden_layers = args.hidden_layer
        self.batch_size = args.batch_size
        self.ip_channels = args.ip_channels
        self.conv_sizes = args.conv_sizes
        self.conv_type = args.conv_type
        self.seq_len = args.seq_len
        self.num_classes = args.num_classes

        # specific arguments for multi-channel
        # number of separate convolutions being carried out in the first layer
        self.num_transforms = args.num_transforms
        # TODO: Figure out best way to handle parameterization for such multiple inputs, or do we do th transformation here ?

        # TODO: Figure out how to implement multi-scale inputs i.e a input in which different channels have different sizes
        # define the data layer
        self.input_layer_1 = tf.placeholder(tf.float32,[self.batch_size, self.seq_len, self.ip_channels])
        self.labels = tf.placeholder(tf.float32,[self.batch_size])

        # first conv-pool layer
        oneD_filter_layer_1 = tf.get_variable('oneD_filter_layer1',[self.conv_sizes[0],self.ip_channels,self.hidden_layers[0]],dtype=tf.float32)
        oneDConv_layer_1 = tf.nn.conv1d(self.input_layer_1,oneD_filter_layer_1,1,'VALID',name='oneDConv_layer1')

        # concatenation
        concat_layer = tf.concat(2,[oneDConv_layer_1])

        # conv-pool layer, halving the input size
        oneD_filter_layer_2 = tf.get_variable('oneD_filter_layer2',[self.conv_sizes[1],self.hidden_layers[0],self.hidden_layers[1]],dtype=tf.float32)
        oneDConv_layer_2 = tf.nn.conv1d(concat_layer,oneD_filter_layer_2,1,'VALID',name='oneDConv_layer2')
        oneDPool_layer2 = tf.nn.max_pool(tf.reshape(oneDConv_layer_2, [self.batch_size,1,-1,self.hidden_layers[1]]),[1,1,2,1],[1,1,1,1],'VALID')

        # fuly connected layer
        fully_layer_3_W = tf.get_variable('fully_layer_3_W',[sum(tf.shape(oneDPool_layer2)[1:]),self.hidden_layers[2]])
        fully_layer_3_b = tf.get_variable('fully_layer_3_b',[self.hidden_layers[2]])
        fully_layer_3 = tf.nn.relu(tf.matmul(tf.reshape(oneDPool_layer2,[self.batch_size,-1]),fully_layer_3_W) + fully_layer_3_b)

        # softmax
        fully_layer_4_W = tf.get_variable('fully_layer_3_W',[self.hidden_layers,self.num_classes])
        fully_layer_4_b = tf.get_variable('fully_layer_3_b', [self.num_classes])
        fully_layer_4 = tf.matmul(fully_layer_3, fully_layer_4_W) + fully_layer_4_b

        # loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(fully_layer_4,self.labels)



class oneDCNNDilated(object):

    def __init__(self,args):
        '''
        class to perform time series classification similar to wavenet paper in terms of using dilated/artorous convs
        '''

        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.num_ip_channels = args.num_ip_channels
        self.len_ip_sequence = args.len_ip_sequence
        self.num_ip_classes = args.num_ip_classes
        self.conv_params = args.conv_params

        # define input and label placeholders
        self.input_layers = tf.placeholder(tf.float32,[self.batch_size,self.len_ip_sequence,self.num_ip_channels])
        self.labels = tf.placeholder(tf.float32,[self.batch_size])

        # first layer of dilated convolution and pooling
        layer1_w = tf.get_variable('layer_1_w',[])
