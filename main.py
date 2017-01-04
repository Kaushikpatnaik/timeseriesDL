import tensorflow as tf
import argparse

import numpy as np
from read_data import *
from train_eval import *
import json
from collections import OrderedDict


class data_args(object):

    dataset = 'mnist'
    batch_size = 64

class dnn_args(object):

    layer_sizes = [64,32,16]
    num_epochs = 2
    lr_rate = 1e-03
    lr_decay = 0.97
    logdir = './logs/dnn'
    model = 'fullDNNNoHistory'

class lstm_args(object):

    cell = 'lstm'
    num_layers = 1
    hidden_units = 32
    num_epochs = 2
    lr_rate = 2e-05
    lr_decay = 0.97
    grad_clip = 5.0
    logdir = './logs/lstm'
    model = 'LSTM'

class cnn_args(object):

    num_layers = 3
    num_epochs = 5
    lr_rate = 1e-03
    lr_decay = 0.97
    logdir = './logs/cnn'
    layer_params = OrderedDict({'3_full': (64), '2_conv': (3,16,[1,1,1],'VALID'), '1_conv': (3,32,[1,1,1],'VALID')})
    model = 'oneDCNN'

class cnn_multi_args(object):

    num_layers = 3
    num_epochs = 5
    lr_rate = 1e-03
    lr_decay = 0.97
    logdir = './logs/cnn'
    layer_params = OrderedDict({'3_full': (64), '2_conv': (3,16,[1,1,1],'VALID'), '1_conv': (3,32,[1,1,1],'VALID')})
    sub_sample = [2,3,4]
    model = 'oneDMultiChannelCNN'


def main():

    args_data = data_args()
    args_model = lstm_args()

    # based on the args_data parameters determine the dataset to be downloaded and split
    train_data, val_data, test_data = get_data_obj(args_data)

    # Training section
    print "Training Dataset Shape: "
    print train_data.shape

    if args_model.model == 'oneDMultiChannelCNN':
        train_data_new = low_pass_and_subsample(train_data)
        val_data_new = low_pass_and_subsample(val_data)
        test_data_new = low_pass_and_subsample(test_data)
    if args_model.model == 'freqCNN':
        train_data_new = freq_transform(train_data)
        val_data_new = freq_transform(val_data)
        test_data_new = freq_transform(test_data)

    batch_train = batchGenerator(train_data, args_data.batch_size, args_data.ip_channels,
                                 args_data.op_channels, args_data.seq_len)
    args_model.max_batches_train = batch_train.get_num_batches()
    args_model.ip_channels = args_data.ip_channels
    args_model.op_channels = args_data.op_channels
    args_model.seq_len = args_data.seq_len
    args_model.batch_size = args_data.batch_size

    # train and return the saved trainable parameters of the model
    train(args_model,batch_train)

    # Validation section
    print "Validation DataSet Shape: "
    print val_data.shape

    batch_val = batchGenerator(val_data,64, args_data.ip_channels, args_data.op_channels, args_data.seq_len)
    args_model.max_batches_val = batch_val.get_num_batches()
    args_model.ip_channels = args_data.ip_channels
    args_model.op_channels = args_data.op_channels
    args_model.seq_len = args_data.seq_len
    args_model.batch_size = 64

    val(args_model,batch_val,"val")


if __name__ == "__main__":
    main()