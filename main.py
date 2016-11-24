import sys
import time

import numpy as np
import models
import layers
import read_data

import tensorflow as tf
import argparse

def train(args,data):

    # Initialize session and graph
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = Model(args, is_training=True, is_inference=False)

        tf.initialize_all_variables().run()

        for i in range(args.num_epochs):
            # TODO: Add parameter for max_max_epochs
            lr_decay = args.lr_decay ** max(i - 10.0, 0.0)
            train_model.assign_lr(session, args.lr_rate * lr_decay)

            # run a complete epoch and return appropriate variables
            train_perplexity = run_epoch(session, train_model, train_model.train_op, batch_train, max_batches_train,
                                         args)
            print 'Epoch %d, Train Perplexity: %.3f' % (i + 1, train_perplexity)

def test(args):

    raise NotImplementedError

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='./data/tinyshakespeare/input.txt',
                        help='data location for all data')
    parser.add_argument('--split_ratio', type=list, default=[0.9, 0.05, 0.05],
                        help='split ratio for train, validation and test')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for data')
    parser.add_argument('--batch_len', type=int, default=1, help='number of time steps to unroll')
    parser.add_argument('--hidden_units', type=list, default=[128,64,32], help='number of hidden units in the cell')
    parser.add_argument('--num_epochs', type=int, default=50, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    args = parser.parse_args()

    # load data
    if args.filename[-3:] == 'zip':
        data = load_zip_data(args.filename)
    elif args.filename[-3:] == 'txt':
        data = load_csv_file(args.filename)
    else:
        raise NotImplementedError("File extension not supported")

    train, val, test = train_test_split(data, args.split_ratio)

    batch_train = BatchGenerator(train, args.batch_size, args.batch_len)
    batch_train.create_batches()
    max_batches_train = batch_train.epoch_size
    # New chars seen in test time will have a problem
    args.data_dim = batch_train.vocab_size

    batch_val = BatchGenerator(val, args.batch_size, args.batch_len)
    batch_val.create_batches()
    max_batches_val = batch_val.epoch_size

    batch_test = BatchGenerator(test, args.batch_size, args.batch_len)
    batch_test.create_batches()
    max_batches_test = batch_test.epoch_size

    print max_batches_train, max_batches_val, max_batches_test

    train(args,train)


if __name__ == "__main__":
    main()