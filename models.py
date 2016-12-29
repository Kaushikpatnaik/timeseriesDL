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

def sequence_loss_by_example(logits, targets, weights, average_across_time=True, scope=None):
  '''
  A simple version of weighted sequence loss measured in sequence
  :param logits:
  :param targets:
  :param weights:
  :param average_across_time:
  :param softmax_loss_function:
  :param scope:
  :return:
  '''
  if len(logits) != len(targets) or len(weights) != len(logits):
    raise ValueError("Lenghts of logits, weights and target must be same "
                     "%d, %d, %d" %len(logits), len(weights), len(targets))

  with tf.variable_scope(scope or "sequence_loss_by_example"):
    sequence_loss_list = []
    for logit, target, weight in zip(logits, targets, weights):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logit,target)
      # tensorflow !!!
      sequence_loss_list.append(loss*weight)
    sequence_loss = tf.add_n(sequence_loss_list)
    if average_across_time:
      total_weight = tf.add_n(weights) + 1e-12
      final_loss = sequence_loss/total_weight
    else:
      final_loss = sequence_loss
    return final_loss

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
        self.mode = args.mode

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
        self.input_layer_x = tf.placeholder(tf.float32,(None,self.ip_channels),'input_layer_x')
        prev_layer = tf.expand_dims(self.input_layer_x, -1)

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

        self.input_layer_y = tf.placeholder(tf.float32,shape=(None,self.op_channels),name='input_layer_y')

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

class oneDCNNDilated(object):

    def __init__(self,args):
        '''
        class to perform time series classification similar to wavenet paper in terms of using dilated/artorous conv
        '''

        raise NotImplementedError


class fullDNNNoHistory(object):

    def __init__(self,args):
        '''
        Function for getting all the parameters

        '''

        self.ip_channels = args.ip_channels
        self.num_layers = len(args.layer_sizes)
        self.op_channels = args.op_channels
        self.layer_sizes = args.layer_sizes
        self.mode = args.mode
        self.init_learn_rate = args.lr_rate

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

        self.input_layer_x = tf.placeholder(dtype=tf.float32,shape=[None,self.ip_channels],name="input_layer_x")

        with tf.variable_scope("layer_0"):
            prev_layer = self.build_single_layer(self.input_layer_x,self.ip_channels,self.layer_sizes[0])

        print [x.name for x in tf.get_collection(tf.GraphKeys.VARIABLES,scope='layer_0')]

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
        self.input_layer_y = tf.placeholder(dtype=tf.float32,shape=[None,self.op_channels],name="input_layer_y")

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
    # TODO: passing multiple flags to the model initiliazation seems hacky, better design or maybe a feedforward function
    def __init__(self, args, is_training, is_inference):
        self.batch_size = args.batch_size
        self.batch_len = args.batch_len
        if is_inference:
          self.batch_size = 1
          self.batch_len = 1
        self.num_layers = args.num_layers
        self.cell = args.cell
        self.hidden_units = args.hidden_units
        self.data_dim = args.data_dim
        self.drop_prob = args.drop_prob
        self.is_training = is_training
        self.is_inference = is_inference

        # define placeholder for data layer
        self.input_layer = tf.placeholder(tf.int32, [self.batch_size, self.batch_len])

        # define weights for data layer (vocab_size to self.hidden_units)
        input_weights = tf.get_variable('input_weights', [self.data_dim, self.hidden_units])

        # define input to LSTM cell [self.batch_size x self.batch_len x self.hidden_units]
        inputs = tf.nn.embedding_lookup(input_weights, self.input_layer)

        # define model based on cell and num_layers
        if self.num_layers ==1:
          self.lstm_layer = LSTM(self.hidden_units,self.drop_prob)
        else:
          cells = [LSTM(self.hidden_units,self.drop_prob)]*self.num_layers
          self.lstm_layer = DeepLSTM(cells)

        outputs = []
        # keep the initial_state accessible (as this will be used for initialization) and state resets with epochs
        #self.initial_state = self.lstm_layer.zero_state(self.batch_size,tf.float32)
        self.initial_state = tf.placeholder(tf.float32,[self.batch_size, self.lstm_layer.state_size])
        state = self.initial_state
        # run the model for multiple time steps
        with tf.variable_scope("RNN"):
          for time in range(self.batch_len):
            if time > 0: tf.get_variable_scope().reuse_variables()
            # pass the inputs, state and weather we are in train/test or inference time (for dropout)
            output, state = self.lstm_layer(inputs[:,time,:], state, (self.is_training or self.is_inference))
            outputs.append(output)

        # for each single input collect the hidden units, then reshape as [self.batch_len x self.batch_size x self.hidden_units]
        output = tf.reshape(tf.concat(1,outputs), [-1,self.hidden_units])
        softmax_w = tf.get_variable('softmax_w', [self.hidden_units, self.data_dim])
        softmax_b = tf.get_variable('softmax_b', [self.data_dim])

        # logits is now of shape [self.batch_size x self.batch_len, self.data_dim]
        self.logits = tf.matmul(output, softmax_w) + softmax_b

        # get probabilities for these logits through softmax (will be needed for sampling)
        self.output_prob = tf.nn.softmax(self.logits)

        # define placeholder for target layer
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.batch_len])

        # sequence loss by example
        # to enable comparision by each and every example the row lengths of logits
        # and targets should be same
        loss = sequence_loss_by_example([self.logits],[tf.reshape(self.targets, [-1])],[tf.ones([self.batch_size*self.batch_len])])
        self.cost = tf.reduce_sum(loss) / self.batch_size / self.batch_len
        self.final_state = state

        if not self.is_training and not self.is_inference:
          return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # clips all gradients, including the weight vectors
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        def assign_lr(self,session,lr_value):
            session.run(tf.assign(self.lr, lr_value))