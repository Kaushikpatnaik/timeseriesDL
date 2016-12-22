import sys
import time
import cPickle

import tensorflow as tf
import argparse

import numpy as np
from read_data import *
from train_eval import *

from tensorflow.examples.tutorials.mnist import input_data

def get_data_obj(args):
    # backblaze_data = blackblazeReader(args)
    # train, val, test = backblaze_data.train_test_split(args.split_ratio)

    # Saved the train, val and test sets for future work, as they take a lot of time to prepare
    # cPickle.dump(train, open('./data/backblaze_' + str(args.drive_model) + '_train.pkl','w'))
    # cPickle.dump(val, open('./data/backblaze_' + str(args.drive_model) + '_val.pkl', 'w'))
    # cPickle.dump(test, open('./data/backblaze_' + str(args.drive_model) + '_test.pkl', 'w'))

    train_data = cPickle.load(open('./data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))
    val_data = cPickle.load(open('./data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))
    test_data = cPickle.load(open('./data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))

    # train_data_raw = open('./data/ElectricDevices_TRAIN','r+')
    # ucr_data = ucrDataReader(train_data_raw,0.8,args.op_channels)
    # train_data, val_data = ucr_data.trainTestSplit()

    # TODO: Figure out the conf details
    parser.add_argument('--op_channels', type= int, default=2, help='number of output classes')

    return train_data, val_data, test_data

def data_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirloc', type=str, default='./data/backblaze',
                        help='data location for all data')
    parser.add_argument('--split_ratio', type=list, default=[0.8, 0.1, 0.1],
                        help='split ratio for train, validation and test')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data')
    parser.add_argument('--drive_model', type=str, default='ST3000DM001', help='drive model for model building')
    parser.add_argument('--hist', type=int, default=5, help='history to use when predicting')
    parser.add_argument('--pred_window', type=int, default=3, help='lookahead to be used for prediction')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')
    args = parser.parse_args()

    return args

def dnn_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_sizes', type=list, default=[64,32,16], help='number of hidden units in the cell')

    parser.add_argument('--num_epochs', type=int, default=5, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    args = parser.parse_args()

    return args

def lstm_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_sizes', type=list, default=[64, 32, 16], help='number of hidden units in the cell')
    parser.add_argument('--num_epochs', type=int, default=5, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    args = parser.parse_args()

    return args

def cnn_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_sizes', type=list, default=[64, 32, 16], help='number of hidden units in the cell')
    parser.add_argument('--num_epochs', type=int, default=5, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    args = parser.parse_args()

    return args

def main():

    args_data = data_args()
    args_dnn = dnn_args()

    train_data, val_data, test_data = get_data_obj(args_data)

    # Training section
    print "Training Dataset Shape: "
    print train_data.shape

    batch_train = batchGenerator(train_data, args.batch_size, args.op_channels)
    batch_train.createBatches()
    args.max_batches_train = batch_train.get_num_batches()
    args.ip_channels = batch_train.get_ip_channels()

    '''
    mnist = input_data.read_data_sets('./data/mnist/input_data', one_hot=True)
    args.ip_channels = 784
    args.max_batches_train = 1000
    '''

    '''
    # Validation section
    print "Validation DataSet Shape: "
    print val_data.shape

    batch_val = batchGenerator(val_data,args.batch_size,args.op_channels)
    batch_val.createBatches()
    args.max_batches_train = batch_val.get_num_batches()
    args.ip_channels = batch_val.get_ip_channels()
    val(args,batch_val,"val")
    '''

    # train and return the saved trainable parameters of the model
    #train(args,mnist)
    train(args,batch_train)

if __name__ == "__main__":
    main()