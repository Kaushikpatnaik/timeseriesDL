'''

File containing implementation of layers not implemented in tensorflow

'''

import tensorflow as tf

def weighted_cross_entropy(weights,logits,labels):
    '''
    Compute the weighted cross entropy provided the weights for each class
    :param weights: tf.constant of shape (num_classes x 1)
    :param logits: pre-softmax final layer output from the NN
    :param labels: one-hot encoding of the labels
    :return: weighted cross entropy loss
    '''

    with tf.name_scope('weighted_cross_entropy'):

        class_weights = tf.constant(weights,dtype=tf.float32)

        weight_per_example = tf.transpose(tf.mul(labels,class_weights))
        reg_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,labels)

        return tf.reduce_mean(tf.mul(weight_per_example,reg_cross_entropy))

def activation_summary(var):
    with tf.name_scope('summary'):
        tensor_name = var.op.name
        mean = tf.reduce_mean(var)
        tf.summary.scalar(tensor_name+'mean',mean)
        std = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar(tensor_name+'std',std)
        tf.summary.scalar(tensor_name+'min',tf.reduce_min(var))
        tf.summary.scalar(tensor_name+'max',tf.reduce_max(var))
        tf.summary.histogram(tensor_name+'histogram',var)

def space_to_batch_1d(input, paddings, block_size, name=None):
    '''
    1d equivalent of space_to_batch function in tensorflow
    Args:
        input: tensor of shape [batch_size, in_width, in_depth]
        paddings: a list of left and right zero padding for the in_width dimension in the input tensor
        block_size: sub-sampling rate in the in_width dimension of the data
        name: function name scope for tensorflow

    Returns:
    Tensor of shape [bath_size*block_size,in_width_pad/block_size,in_depth]
    '''

    with tf.name_scope(name, "space_to_batch_1d") as name:

        if block_size < 1:
            raise ValueError("block_size must be greater than 1")

        ip_shape = tf.shape(input)
        padded = tf.pad(input,paddings)
        reshaped = tf.reshape(padded,[-1,block_size,ip_shape[2]])
        permuted = tf.transpose(reshaped,[1,0,2])
        return tf.reshape(permuted,[ip_shape[0]*block_size,-1,ip_shape[2]])

def batch_to_space_1d(input, crop, block_size, name=None):

    with tf.name_scope(name, "batch_to_space_1d") as name:

        if block_size < 1:
            raise ValueError("block_size must be greater than 1")

        ip_shape = tf.shape(input)
        batch_size = ip_shape[0]/block_size
        reshaped = tf.reshape(input,[block_size,-1,ip_shape[2]])
        permuted = tf.transpose(reshaped,[1,0,2])
        reshaped2 = tf.reshape(permuted, [batch_size,-1,ip_shape[2]])
        return tf.crop(reshaped2,crop)

def artrous_conv1d(value, filters, rate, padding, name=None):
    '''
    1d equivalent of the tensorflow artrous 2d convolution

    Args:
        value: input 3-D tensor with shape [batch_size, in_width, ip_channels]
        filters: convolution filter with shape [filter_width, ip_channels, op_channels]
        rate: rate of dilation, 1 is equivalent to normal convolution
        padding: string, "SAME" or "VALID" only
        name: function scoping name for tensorflow

    Returns:
    Tensor of shape [batch_size, new_width, op_channels]
    '''
    with tf.name_scope(name, "artrous_conv1d") as name:
        if rate == 1:
            # perform normal convolution
            return tf.nn.conv1d(value,filters,stride=1,padding=padding)

        elif rate > 1:
            # determine padding based on padding choice
            if padding == "SAME":
                # based on the size of the kernel add zeroes to the image
                filter_width = tf.shape(filters)[0]
                overall_pad = filter_width + filter_width*(rate-1)
                pad_left = overall_pad//2
                pad_right = overall_pad - pad_left

            else:
                pad_left = 0
                pad_right = 0

            # check optimality with the rate provided
            check_width = pad_left + tf.shape(value) + pad_right
            pad_right_extra = (rate - check_width % rate) % rate
            pad_right += pad_right_extra

            # reshape value tensor based on type of padding and rate
            new_value = space_to_batch_1d(value,[pad_left,pad_right],rate)

            # perform 1d convolution
            conv_new_value = tf.nn.conv1d(new_value,filters,stride=1,padding='VALID')

            # re-shape the output to output size
            conv_value = batch_to_space_1d(conv_new_value,[0,pad_right_extra],rate)

            return conv_value

        else:
            raise ValueError("Rate must be >= 1")

class lstmLayer(object):
    '''A single LSTM unit with one hidden layer'''

    def __init__(self, hidden_units, offset_bias=1.0):
        '''
        Initialize the LSTM with given number of hidden layers and the offset bias
        :param hidden_units: number of hidden cells in the LSTM
        :param offset_bias: the bias is usually kept as 1.0 initially for ?? TODO: find out reason
        :param drop_prob: dropout probability
        '''

        self.hidden_units = hidden_units
        self.offset_bias = offset_bias
        self.state_size = 2 * self.hidden_units

    def __call__(self, input_data, state, scope=None):
        '''
        Take in input_data and update the hidden unit and the cell state
        :param input_data: data for the current time step
        :param state: previous cell state
        :param scope: scope within which the variables exist
        :return: new cell state and output concated
        '''
        with tf.variable_scope(scope or type(self).__name__):
            # Recurrent weights are always of size hidden_layers*hidden_layers
            # Input to hidden are always of size vocab_size*hidden_layers
            # Cell state and output are of size batch_size * hidden_units
            # Input_data is of size batch_size * vocab

            # separate the cell state from output
            c, h = tf.split(1, 2, state)

            # Overall there are four set of input to hidden weights, and four set of hidden to hidden weights
            # All of them can be processed together as part of one array operation or by creating a function and
            # scoping the results appropriately
            # TODO: Kaushik Add initialization schemes
            def sum_inputs(input_data, h, scope):
                with tf.variable_scope(scope):
                    ip2hiddenW = tf.get_variable('ip2hidden',
                                                 shape=[input_data.get_shape()[1], self.hidden_units],
                                                 dtype=tf.float32,initializer=tf.random_uniform_initializer())
                    hidden2hiddenW = tf.get_variable('hidden2hidden',
                                                     shape=[self.hidden_units, self.hidden_units],
                                                     dtype=tf.float32,initializer=tf.random_uniform_initializer())
                    biasW = tf.get_variable('biasW', shape=[self.hidden_units],
                                            dtype=tf.float32,initializer=tf.constant_initializer(0.0))
                    ip2hidden = tf.matmul(input_data, ip2hiddenW) + biasW
                    hidden2hidden = tf.matmul(h, hidden2hiddenW) + biasW
                    return ip2hidden + hidden2hidden

            ip_gate = sum_inputs(input_data, h, 'input_gate')
            ip_transform = sum_inputs(input_data, h, 'input_transform')
            forget_gate = sum_inputs(input_data, h, 'forget_gate')
            output_gate = sum_inputs(input_data, h, 'output_gate')

            new_c = c * tf.sigmoid(forget_gate + self.offset_bias) + tf.sigmoid(ip_transform) * tf.tanh(ip_gate)
            new_h = tf.tanh(new_c) * tf.sigmoid(output_gate)

            return new_h, tf.concat(1, [new_c, new_h])

class DeepLSTM(object):
    '''A DeepLSTM unit composed of multiple LSTM units'''

    def __init__(self, cells):
        '''
        :param cell: list of LSTM cells that are to be stacked
        :param drop_porb: layerwise regularization using dropout
        '''
        self.cells = cells
        self.state_size = sum([cell.state_size for cell in cells])

    def __call__(self, input_data, state, scope=None):
        '''
        Go through multiple layers of the cells and return the final output and all the cell states
        :param input_data: data for the current time step
        :param state: previous cell states for all the layers
        :param is_training: boolean flag capturing whether training is being done or not
        :param scope: scope within which the operation will occur
        :return: new cell states and final output layer
        '''
        with tf.variable_scope(scope or type(self).__name__):
            # with multiple layers we need to iterate through each layer, and update its weights and cell states
            # to ensure no collision among weights, we should scope within the layer loop also
            new_states = []
            curr_pos = 0
            curr_input = input_data
            for i, cell in enumerate(self.cells):
                with tf.variable_scope("Cell_" + str(i)):
                    curr_state = tf.slice(state, [0, curr_pos], [-1, cell.state_size])
                    curr_pos += cell.state_size
                    # hidden unit is propagated as the input_data
                    curr_input, new_state = cell(curr_input, curr_state)
                    new_states.append(new_state)
            return curr_input, tf.concat(1, new_states)

    def zero_state(self, batch_size, dtype):
        '''
        return a zero shaped vector (used in initialization schemes)
        :param batch_size: size of batch
        :param dtype: data type of the batch
        :return: a 2D tensor of shape [batch_size x state_size]
        '''
        initial_state = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
        return initial_state

# TODO: doing maxpooling over 1D convolution output
def maxPool1D():
    raise NotImplementedError