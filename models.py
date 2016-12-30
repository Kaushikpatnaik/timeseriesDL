'''
Using the resnet implementation by Google to be an inspiration for the code
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from layers import *


def _activation_summary(var):
    with tf.name_scope('summary'):
        tensor_name = var.op.name
        mean = tf.reduce_mean(var)
        tf.scalar_summary(tensor_name+'mean',mean)
        std = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.scalar_summary(tensor_name+'std',std)
        tf.scalar_summary(tensor_name+'min',tf.reduce_min(var))
        tf.scalar_summary(tensor_name+'max',tf.reduce_max(var))
        tf.histogram_summary(tensor_name+'histogram',var)

class oneDCNN(object):

    def __init__(self,args):
        '''
        Gather all the parameters required to define and initialize a oneDNN
        Args:
            args: model arguments collected from main
        '''

        self.num_layers = args.num_layers
        self.layer_params = args.layer_params
        self.init_lr_rate = args.lr_rate
        self.ip_channels = args.ip_channels
        self.op_channels = args.op_channels
        self.seq_len = args.seq_len
        self.mode = args.mode
        self.batch_size = args.batch_size

    def build_graph(self):

        self._build_model()

        if self.mode == 'train':
            self._add_train_nodes()
        self.summaries = tf.merge_all_summaries()

    def build_cnn_layer(self,prev_layer,kernel_size,stride,padding,scope_name):

        with tf.variable_scope(scope_name):
            kernel = tf.get_variable('conv_weight',shape=kernel_size,dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(0,0.001),
                                     regularizer=tf.contrib.layers.l2_regularizer(0.1))
            conv_op = tf.nn.conv1d(prev_layer,kernel,stride,padding)
            bias = tf.get_variable('conv_bias',shape=kernel_size[-1],dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            nonlinear_op = tf.nn.relu(tf.nn.bias_add(conv_op,bias))
            _activation_summary(nonlinear_op)

        return nonlinear_op

    def build_cnn_pool_layer(self,prev_layer,kernel_size,stride,padding,pool_size,scope_name):

        raise NotImplementedError

    def build_full_layer(self,prev_layer, ip_size, op_size,scope_name):

        # TODO: pass mean and std dev of initialization as parameters
        # TODO: pass regularization constanst as parameter
        with tf.variable_scope(scope_name):
            layer_w = tf.get_variable('layer_w', [ip_size, op_size], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(0,0.001),
                                      regularizer=tf.contrib.layers.l2_regularizer(0.1))
            layer_b = tf.get_variable('layer_b', [op_size], dtype=tf.float32,
                                      initializer= tf.constant_initializer(0.0))
            local = tf.nn.relu(tf.matmul(prev_layer, layer_w) + layer_b)
            _activation_summary(local)

        return local

    def _build_model(self):

        # TODO: None batch_size propagation becomes complicated due to reshaping op later on
        self.input_layer_x = tf.placeholder(tf.float32,(self.batch_size,self.seq_len,self.ip_channels),'input_layer_x')
        prev_layer = self.input_layer_x

        print self.layer_params

        # iteratively build the layers
        # TODO: Check with tensorboard if this is being done accurately
        for i in range(self.num_layers):
            if self.layer_params.keys()[i].split('_')[1] == 'conv':

                print "Building conv layer for "+str(i)
                kernel_width, kernel_op_channel, stride, padding = self.layer_params.values()[i]
                kernel_ip_channel = prev_layer.get_shape()[-1]
                kernel_size = [kernel_width,kernel_ip_channel,kernel_op_channel]
                prev_layer = self.build_cnn_layer(prev_layer,kernel_size,stride,padding,'conv_'+str(i))

            elif self.layer_params.keys()[i].split('_')[1] == 'full':

                print "Building full layer for "+str(i)

                if self.layer_params.keys()[i-1].split('_')[1] != 'full':
                    row, col, channel = prev_layer.get_shape()
                    prev_layer = tf.reshape(prev_layer,[-1,int(col*channel)])
                    ip_size = col*channel
                    op_size = self.layer_params.values()[i]
                    prev_layer = self.build_full_layer(prev_layer,ip_size,op_size,'full_'+str(i))
                else:
                    op_size = self.layer_params.values()[i]
                    ip_size = prev_layer.get_shape()[-1]
                    prev_layer = self.build_full_layer(prev_layer,ip_size,op_size,'full_'+str(i))

            elif self.layer_params.keys()[i].split('_')[1] == 'conv_pool':

                print "Building conv_pool layer for "+str(i)
                kernel_width, kernel_op_channel, stride, padding, pool_size = self.layer_params.values()[i]
                kernel_ip_channel = prev_layer.get_shape()[-1]
                kernel_size = [kernel_width,kernel_ip_channel,kernel_op_channel]
                prev_layer = self.build_cnn_pool_layer(prev_layer,kernel_size,stride,padding,pool_size,'conv_pool_'+str(i))

            else:
                raise ValueError("layer specified has not been implemented")

        # need to flatten the output if the final layer is not a fully connected layer
        final_layer = prev_layer
        if self.layer_params.keys()[-1].split('_')[1] != 'full':
            row, col, channel = final_layer.get_shape()
            final_layer = tf.reshape(final_layer,[-1,int(col*channel)])

        # softmax output from final layer
        softmax_w = tf.get_variable('softmax_w',[np.prod(final_layer.get_shape()[1:]),self.op_channels],dtype=tf.float32,
                                    initializer= tf.truncated_normal_initializer(0,0.001),
                                    regularizer=tf.contrib.layers.l2_regularizer(0.1))
        softmax_b = tf.get_variable('softmax_b',[self.op_channels],dtype=tf.float32)
        self.output = tf.matmul(final_layer,softmax_w) + softmax_b
        self.output_prob = tf.nn.softmax(self.output)
        _activation_summary(self.output_prob)

    def _add_train_nodes(self):

        self.input_layer_y = tf.placeholder(tf.float32,shape=(self.batch_size,self.op_channels),name='input_layer_y')

        # compute cross entropy loss
        cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output,self.input_layer_y))
        tf.scalar_summary('cross_entropy_loss',cross_loss)

        # gather regularization terms and add them to the total loss
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_scale = 0.1
        reg_losses = reg_scale * tf.add_n(reg_var)
        tf.scalar_summary('regularization_loss',reg_losses)

        self.cost = cross_loss + reg_losses
        tf.scalar_summary('total_loss',self.cost)

        # learning rate
        self.lrn_rate = tf.Variable(self.init_lr_rate,trainable=False)

        # get trainable variables, define optimizer, compute gradients and apply them
        trainable_params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        gradients = optimizer.compute_gradients(self.cost,trainable_params)

        # histogram_summaries for weights and gradients
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name+'/gradient',grad)

        self.train_op = optimizer.apply_gradients(gradients)

    def assign_lr(self,sess,new_value):
        sess.run(tf.assign(self.lrn_rate,new_value))

class oneDMultiChannelCNN(object):

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

        raise NotImplementedError


class fullDNNNoHistory(object):

    def __init__(self,args):
        '''
        Function for getting all the parameters

        '''

        self.ip_channels = args.ip_channels
        self.seq_len = args.seq_len
        self.num_layers = len(args.layer_sizes)
        self.op_channels = args.op_channels
        self.layer_sizes = args.layer_sizes
        self.mode = args.mode
        self.init_learn_rate = args.lr_rate
        self.batch_size = args.batch_size

    def build_graph(self):

        self._build_model()

        if self.mode == 'train':
            self._add_train_nodes()
        self.summaries = tf.merge_all_summaries()

    def build_single_layer(self,prev_layer, ip_size, op_size):

        # TODO: pass mean and std dev of initialization as parameters
        # TODO: pass regularization constanst as parameter
        # Build the first layer of weights, build the next ones iteratively
        layer_w = tf.get_variable('layer_w', [ip_size, op_size], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0,0.001),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))
        layer_b = tf.get_variable('layer_b', [op_size], dtype=tf.float32,
                                  initializer= tf.truncated_normal_initializer(0,0.001))
        local = tf.nn.relu(tf.matmul(prev_layer, layer_w) + layer_b)
        _activation_summary(local)

        return local

    def _build_model(self):
        '''
        Initialize and define the model to be use for computation
        Returns:

        '''

        self.input_layer_x = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.seq_len,self.ip_channels],name="input_layer_x")
        input_layer = tf.squeeze(tf.reshape(self.input_layer_x,[self.batch_size,self.seq_len*self.ip_channels,1]),squeeze_dims=[2])

        with tf.variable_scope("layer_0"):
            prev_layer = self.build_single_layer(input_layer,self.ip_channels*self.seq_len,self.layer_sizes[0])

        # Iterate over layers size with proper scope to define the higher layers
        for i in range(1,self.num_layers):

            curr_layer_size = self.layer_sizes[i]
            prev_layer_size = self.layer_sizes[i-1]

            with tf.variable_scope("layer_"+str(i)):
                prev_layer = self.build_single_layer(prev_layer,prev_layer_size,curr_layer_size)

        softmax_w = tf.get_variable('softmax_w',[self.layer_sizes[-1],self.op_channels],dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.1))
        softmax_b = tf.get_variable('softmax_b',[self.op_channels],dtype=tf.float32)
        self.output = tf.matmul(prev_layer,softmax_w) + softmax_b
        self.output_prob = tf.nn.softmax(self.output)
        _activation_summary(self.output_prob)


    def _add_train_nodes(self):
        '''
        Define the loss layer, learning rate and optimizer
        Returns:

        '''
        self.input_layer_y = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.op_channels],name="input_layer_y")

        # gather loss from regularization variables
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.1

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output,self.input_layer_y)) + reg_constant*tf.add_n(reg_losses)
        tf.scalar_summary("loss",self.cost)

        self.lrn_rate = tf.Variable(self.init_learn_rate,trainable=False,dtype=tf.float32)
        tf.scalar_summary('learning_rate',self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        grads_vars = optimizer.compute_gradients(self.cost,trainable_variables)

        # histogram_summaries for weights and gradients
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads_vars:
            if grad is not None:
                tf.histogram_summary(var.op.name+'/gradient',grad)

        self.train_op = optimizer.apply_gradients(grads_vars)

    def assign_lr(self, session, new_lr):
        session.run(tf.assign(self.lrn_rate, new_lr))


# TODO: test out the LSTM on data
class LSTM(object):
    '''Class defining the overall model based on layers.py'''
    def __init__(self, args):

        self.seq_len = args.seq_len
        self.num_layers = args.num_layers
        self.cell = args.cell
        self.hidden_units = args.hidden_units
        self.ip_channels = args.ip_channels
        self.op_classes = args.op_channels
        self.mode = args.mode
        self.init_lr = args.lr_rate
        self.grad_clip = args.grad_clip
        self.batch_size = args.batch_size

    def build_graph(self):

        self._build_model()

        if self.mode == 'train':
            self._add_train_nodes()
        self.summaries = tf.merge_all_summaries()

    def _build_model(self):

        # define placeholder for data layer
        self.input_layer_x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_len, self.ip_channels],name='input_layer_x')

        # define model based on cell and num_layers
        if self.num_layers ==1:
          self.lstm_layer = lstmLayer(self.hidden_units)
        else:
          cells = [lstmLayer(self.hidden_units)]*self.num_layers
          self.lstm_layer = DeepLSTM(cells)

        # TODO: Think about the need for statefullness in this problem scenario
        self.initial_state = tf.zeros([self.batch_size,self.lstm_layer.state_size],tf.float32)
        #self.initial_state = tf.placeholder(tf.float32,[None, self.lstm_layer.state_size])

        state = self.initial_state
        output = tf.zeros([self.batch_size,self.hidden_units],dtype=tf.float32)
        # run the model for multiple time steps
        with tf.variable_scope("RNN"):
          for time in range(self.seq_len):
            if time > 0: tf.get_variable_scope().reuse_variables()
            # pass the inputs, state and weather we are in train/test or inference time (for dropout)
            output, state = self.lstm_layer(self.input_layer_x[:,time,:], state)

        self.final_state = state
        self.final_output = output

        # for each single input collect the hidden units, then reshape as [self.seq_len x self.batch_size x self.hidden_units]
        softmax_w = tf.get_variable('softmax_w', [self.hidden_units, self.op_classes],dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer())
        softmax_b = tf.get_variable('softmax_b', [self.op_classes],dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer())

        # logits is now of shape [self.batch_size x self.seq_len, self.data_dim]
        self.logits = tf.matmul(self.final_output, softmax_w) + softmax_b

        # get probabilities for these logits through softmax (will be needed for sampling)
        self.output_prob = tf.nn.softmax(self.logits)

    def _add_train_nodes(self):

        # define placeholder for target layer
        self.input_layer_y = tf.placeholder(tf.float32, [self.batch_size,self.op_classes],name='input_layer_y')

        # sequence loss by example
        # TODO: Implement proper loss function for encoder like structure of LSTM
        # TODO: Add regularization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,self.input_layer_y))
        tf.scalar_summary("loss",self.cost)

        self.lr = tf.Variable(self.init_lr, trainable=False)
        trainable_variables = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_vars = optimizer.compute_gradients(self.cost,trainable_variables)

        # histogram_summaries for weights and gradients
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads_vars:
            if grad is not None:
                tf.histogram_summary(var.op.name+'/gradient',grad)

        # TODO: Think about how gradient clipping is implemented, cross check
        grads, _ = tf.clip_by_global_norm([grad for (grad,var) in grads_vars], self.grad_clip)
        self.train_op = optimizer.apply_gradients(grads_vars)

    def assign_lr(self,session,lr_value):
        session.run(tf.assign(self.lr, lr_value))