import cPickle

import tensorflow as tf
import argparse

import numpy as np
from read_data import *
from train_eval import *
import json
from collections import OrderedDict

from tensorflow.examples.tutorials.mnist import input_data

def get_data_obj(args):
    '''
    Reading the data and flatting the sequence. i.e each row is seq_len (history) * num of channels (features)
    Args:
        args:

    Returns:

    '''

    # TODO: determine the dataset num classes automatically
    if args.dataset == 'backblaze':

        # These options only work if you are creating a new dataset
        # TODO: Need to find better way to do this
        args.drive_model = 'ST3000DM001'
        args.hist = 5
        args.pred_window = 3
        args.op_channels = 2


        # backblaze_data = blackblazeReader(args)
        # train, val, test = backblaze_data.train_test_split(args.split_ratio)

        # Saved the train, val and test sets for future work, as they take a lot of time to prepare
        # cPickle.dump(train, open('./data/backblaze_' + str(args.drive_model) + '_train.pkl','w'))
        # cPickle.dump(val, open('./data/backblaze_' + str(args.drive_model) + '_val.pkl', 'w'))
        # cPickle.dump(test, open('./data/backblaze_' + str(args.drive_model) + '_test.pkl', 'w'))

        train_data = cPickle.load(open('./data/backblaze/processed_data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))
        val_data = cPickle.load(open('./data/processed_data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))
        test_data = cPickle.load(open('./data/processed_data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))

    elif args.dataset == 'electric':

        args.op_channels = 7
        args.seq_len = 96
        args.ip_channels = 1
        train_data_raw = open('./data/ucr/ElectricDevices_TRAIN','r+')
        ucr_data = ucrDataReader(train_data_raw,args.split_ratio,args.op_channels)
        train_data, val_data, test_data = ucr_data.trainTestSplit()

    elif args.dataset == "mnist":

        mnist = input_data.read_data_sets('./data/mnist/input_data', one_hot=True)
        args.op_channels = 10
        args.ip_channels = 28
        args.seq_len = 28
        train_data = np.hstack((mnist.train.images, mnist.train.labels))
        val_data = np.hstack((mnist.validation.images, mnist.validation.labels))
        test_data = np.hstack((mnist.test.images, mnist.test.labels))

    else:
        raise ValueError("Dataset option provided does not exist")

    return train_data, val_data, test_data

def data_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',help='data location for all data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data')
    args = parser.parse_args()

    return args

def dnn_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_sizes', type=list, default=[64,32,16], help='number of hidden units in the cell')
    parser.add_argument('--num_epochs', type=int, default=5, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--drop_prob', type=float, default=0, help='dropout probability')
    parser.add_argument('--logdir', type=str, default='./logs/dnn', help='log directory')
    args = parser.parse_args()
    args.model = 'fullDNNNoHistory'

    return args

def lstm_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=str, default='lstm', help='the cell type to use, currently only LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='depth of hidden units in the model')
    parser.add_argument('--hidden_units', type=int, default=32, help='number of hidden units in the cell')
    parser.add_argument('--num_epochs', type=int, default=2, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='clip gradients at this value')
    parser.add_argument('--logdir', type=str, default='./logs/lstm', help='log directory')

    args = parser.parse_args()
    args.model = 'LSTM'

    return args

def cnn_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
    # TODO: Change to layer params and think about method for padding, kernel size and stride
    parser.add_argument('--num_epochs', type=int, default=5, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--init_std_dev', type=float, default=0.001, help='std. deviation to be used for truncated normalized initialization')
    parser.add_argument('--reg_scale', type=float, default=0.01, help='regularization scaling factor to be used')
    parser.add_argument('--logdir', type=str, default='./logs/cnn', help='log directory')
    args = parser.parse_args()
    # Note: V0.10 has conv1d not part of API, so the implementation is hacky at best
    args.layer_params = OrderedDict({'3_full': (64), '2_conv': (3,16,1,'VALID'), '1_conv': (3,32,1,'VALID')})
    args.model = 'oneDCNN'

    return args

def main():

    args_data = data_args()
    args_model = lstm_args()

    # based on the args_data parameters determine the dataset to be downloaded and split
    train_data, val_data, test_data = get_data_obj(args_data)

    # Training section
    print "Training Dataset Shape: "
    print train_data.shape

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
    args_model.max_batches_train = batch_val.get_num_batches()
    args_model.ip_channels = args_data.ip_channels
    args_model.op_channels = args_data.op_channels
    args_model.seq_len = args_data.seq_len
    args_model.batch_size = 64

    val(args_model,batch_val,"val")



if __name__ == "__main__":
    main()