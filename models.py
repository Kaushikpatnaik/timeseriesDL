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

    def _build_model(self):
        '''
        Initialize and define the model to be use for computation
        Returns:

        '''

        def build_single_layer(prev_layer,ip_size,op_size):
            # Build the first layer of weights, build the next ones iteratively
            layer_w = tf.get_variable('layer_w', [ip_size, op_size], dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.1))
            layer_b = tf.get_variable('layer_b', [op_size], dtype=tf.float32)
            local = tf.nn.relu(tf.matmul(prev_layer, layer_w) + layer_b)
            return local

        self.input_layer_x = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.ip_channels],name="input_layer_x")

        with tf.variable_scope("layer_0"):
            prev_layer = build_single_layer(self.input_layer_x,self.ip_channels,self.layer_sizes[0])

        print [x.name for x in tf.get_collection(tf.GraphKeys.VARIABLES,scope='layer_0')]

        # Iterate over layers size with proper scope to define the higher layers
        for i in range(1,self.num_layers):

            curr_layer_size = self.layer_sizes[i]
            prev_layer_size = self.layer_sizes[i-1]

            with tf.variable_scope("layer_"+str(i)):
                prev_layer = build_single_layer(prev_layer,prev_layer_size,curr_layer_size)

        softmax_w = tf.get_variable('softmax_w',[self.layer_sizes[-1],self.op_channels],dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(0.1))
        softmax_b = tf.get_variable('softmax_b',[self.op_channels],dtype=tf.float32)
        self.output = tf.matmul(prev_layer,softmax_w) + softmax_b
        self.output_prob = tf.nn.softmax(self.output)

        tf.histogram_summary('op_prob',self.output_prob)

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
        grads = tf.gradients(self.cost, trainable_variables)

        optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    def assign_lr(self, session, new_lr):
        session.run(tf.assign(self.lrn_rate, new_lr))


class fullDNNWithHistory(object):

    def __init__(self,args):
        '''
        Function for getting all the parameters

        '''

        raise NotImplementedError


from layers import *
from tensorflow.python.ops import math_ops
import tensorflow as tf

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
    sequence_loss = math_ops.add_n(sequence_loss_list)
    if average_across_time:
      total_weight = math_ops.add_n(weights) + 1e-12
      final_loss = sequence_loss/total_weight
    else:
      final_loss = sequence_loss
    return final_loss

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