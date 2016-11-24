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

        raise NotImplementedError



class oneDCNNDilated(object):

    def __init__(self,args):
        '''
        class to perform time series classification similar to wavenet paper in terms of using dilated/artorous conv
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

        raise NotImplementedError


class fullDNNNoHistory(object):

    def __init__(self,args):
        '''
        Function for getting all the parameters

        '''

        self.batch_size = args.batch_size
        self.ip_channels = args.ip_channels
        self.num_layers = args.num_layers
        self.op_channels = args.op_channels
        self.hidden_unit = args.hidden_unit
        self.layer_sizes = args.layer_sizes
        self.mode = args.mode
        self.learn_rate = args.learn_rate


    def build_graph(self):

        self._build_model()

        if self.mode == 'train':
            self._add_train_nodes()
        self.summaries = tf.merge_all_summaries()


    def _build_model(self,args):
        '''
        Initialize and define the model to be use for computation
        Returns:

        '''

        self.input_layer_x = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.ip_channels],name="input_layer_x")
        self.input_layer_y = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,1],name="'input_layer_y")

        # Build the first layer of weights, build the next ones iteratively
        with tf.variable_scope("layer_0"):
            layer_w = tf.get_variable('layer_w', [self.ip_channels,self.layer_sizes[0]],dtype=tf.float32)
            layer_b = tf.get_variable('layer_b', [self.ip_channels,1],dtype=tf.float32)
            layer_h = tf.matmul(layer_w,self.input_layer_x) + layer_b

        # Iterate over layers size with proper scope to define the higher layers
        for i in range(1,len(self.num_layers)):

            curr_layer_size = self.layer_sizes[i]
            prev_layer_size = self.layer_sizes[i-1]

            with tf.variable_scope("layer_"+str(i-1),reuse=True):
                prev_layer = tf.get_variable(layer_h)

            with tf.variable_scope("layer_"+str(i)):
                layer_w = tf.get_variable('layer_w',[prev_layer_size,curr_layer_size],dtype=tf.float32)
                layer_b = tf.get_variable('layer_b',[prev_layer_size,1],dtype=tf.float32)
                layer_h = tf.matmul(layer_w,prev_layer) + layer_b

        # final layer with prediction of class
        with tf.variable_scope("layer_"+str(len(self.num_layers)-1)):
        final_hidden_layer = tf.get_variable(layer_h)

        softmax_w = tf.get_variable('softmax_w',[self.layer_sizes[-1],self.op_channels],dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b',[self.layer_sizes[-1],self.op_channels],dtype=tf.float32)
        self.output = tf.matmul(softmax_w,final_hidden_layer) + softmax_b

        self.output_prob = tf.nn.softmax(output,name="output_layer")

        tf.scalar_summary(self.output_prob,'op_prob')


    def _add_train_nodes(self):
        '''
        Define the loss layer, learning rate and optimizer
        Returns:

        '''

        self.input_layer_y = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,1],name="'input_layer_y")

        self.cost = tf.nn.softmax_cross_entropy_with_logits(self.output,self.input_layer_x)
        tf.scalar_summary(self.cost,"loss")

        self.lrn_rate = tf.constant(self.learn_rate,tf.float32)
        tf.scalar_summary(self.lrn_rate,'learning_rate')

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


class fullDNNWithHistory(object):

    def __init__(self,args):
        '''
        Function for getting all the parameters

        '''

        self.batch_size = args.batch_size
        self.ip_channels = args.ip_channels
        self.num_layers = args.num_layers
        self.op_channels = args.op_channels
        self.hidden_unit = args.hidden_unit
        self.layer_sizes = args.layer_sizes
        self.mode = args.mode
        self.learn_rate = args.learn_rate


    def build_graph(self):

        self._build_model()

        if self.mode == 'train':
            self._add_train_nodes()
        self.summaries = tf.merge_all_summaries()


    def _build_model(self,args):
        '''
        Initialize and define the model to be use for computation
        Returns:

        '''

        self.input_layer_x = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.ip_channels],name="input_layer_x")
        self.input_layer_y = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,1],name="'input_layer_y")

        # Build the first layer of weights, build the next ones iteratively
        with tf.variable_scope("layer_0"):
            layer_w = tf.get_variable('layer_w', [self.ip_channels,self.layer_sizes[0]],dtype=tf.float32)
            layer_b = tf.get_variable('layer_b', [self.ip_channels,1],dtype=tf.float32)
            layer_h = tf.matmul(layer_w,self.input_layer_x) + layer_b

        # Iterate over layers size with proper scope to define the higher layers
        for i in range(1,len(self.num_layers)):

            curr_layer_size = self.layer_sizes[i]
            prev_layer_size = self.layer_sizes[i-1]

            with tf.variable_scope("layer_"+str(i-1),reuse=True):
                prev_layer = tf.get_variable(layer_h)

            with tf.variable_scope("layer_"+str(i)):
                layer_w = tf.get_variable('layer_w',[prev_layer_size,curr_layer_size],dtype=tf.float32)
                layer_b = tf.get_variable('layer_b',[prev_layer_size,1],dtype=tf.float32)
                layer_h = tf.matmul(layer_w,prev_layer) + layer_b

        # final layer with prediction of class
        with tf.variable_scope("layer_"+str(len(self.num_layers)-1)):
        final_hidden_layer = tf.get_variable(layer_h)

        softmax_w = tf.get_variable('softmax_w',[self.layer_sizes[-1],self.op_channels],dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b',[self.layer_sizes[-1],self.op_channels],dtype=tf.float32)
        self.output = tf.matmul(softmax_w,final_hidden_layer) + softmax_b

        self.output_prob = tf.nn.softmax(output,name="output_layer")

        tf.scalar_summary(self.output_prob,'op_prob')


    def _add_train_nodes(self):
        '''
        Define the loss layer, learning rate and optimizer
        Returns:

        '''

        self.input_layer_y = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,1],name="'input_layer_y")

        self.cost = tf.nn.softmax_cross_entropy_with_logits(self.output,self.input_layer_x)
        tf.scalar_summary(self.cost,"loss")

        self.lrn_rate = tf.constant(self.learn_rate,tf.float32)
        tf.scalar_summary(self.lrn_rate,'learning_rate')

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))