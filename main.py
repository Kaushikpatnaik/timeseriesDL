import sys
import time

import tensorflow as tf
import argparse

import numpy as np
from models import *
from read_data import *
from layers import *

def run_epoch(session, model, data, args, max_batches):
  '''
  Run the model under given session for max_batches based on args
  :param model: model on which the operations take place
  :param session: session for tensorflow
  :param train_op: training output variable name, pass as tf.no_op() for validation and testing
  :param data: train, validation or testing data
  :param max_batches: maximum number of batches that can be called
  :param args: arguments provided by user in main
  :return: perplexity
  '''

  # to run a session you need the list of tensors/graph nodes and the feed dict
  # for us its the cost, final_state, and optimizer
  # you feed in the (x,y) pairs, and you also propagate the state across the batches
  tot_cost = 0.0
  start_time = time.time()
  iters = 0

  for i in range(max_batches):
    x, y = data.next()
    cur_cost, curr_state, _ = session.run([model.cost,model.output_prob,model.train_op],
                feed_dict={model.input_layer_x: x, model.input_layer_y: y})
    tot_cost += cur_cost
    state = curr_state
    iters += args.batch_len

  return np.exp(tot_cost/iters)

def train(args,batch_train):

    # Initialize session and graph
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope("model", reuse=None, initializer=initializer):

            args.mode = 'train'
            if args.model_opt == 'fullDNNNoHistory':
                train_model = fullDNNNoHistory(args)
            elif args.model_opt == 'fullDNNWithHistory':
                train_model = fullDNNWithHistory(args)
            else:
                raise NotImplementedError

        tf.initialize_all_variables().run()

        for i in range(args.num_epochs):
            # TODO: Add parameter for max_max_epochs
            lr_decay = args.lr_decay ** max(i - 10.0, 0.0)
            train_model.assign_lr(session, args.lr_rate * lr_decay)

            # run a complete epoch and return appropriate variables
            train_perplexity = run_epoch(session, train_model, batch_train, args, args.max_batches_train)
            print 'Epoch %d, Train Perplexity: %.3f' % (i + 1, train_perplexity)

def test(args):

    raise NotImplementedError

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, default='./data/tinyshakespeare/input.txt',
                        help='data location for all data')
    parser.add_argument('--split_ratio', type=list, default=[0.8, 0.1, 0.1],
                        help='split ratio for train, validation and test')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for data')
    parser.add_argument('--batch_len', type=int, default=1, help='number of time steps to unroll')
    parser.add_argument('--layer_sizes', type=list, default=[128,64,32], help='number of hidden units in the cell')
    parser.add_argument('--num_epochs', type=int, default=50, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--drive_model', type=str, default='ST3000DM001', help='drive model for model building')
    parser.add_argument('--hist', type=int, default=5, help='history to use when predicting')
    parser.add_argument('--pred_window', type=int, default=3, help='lookahead to be used for prediction')
    parser.add_argument('--model_opt', type=str, default='fullDNNNoHistory',help='model to be tested')


    args = parser.parse_args()

    backblaze_data = blackblazeReader(args)

    train, val, test = backblaze_data.train_test_split(args.split_ratio)

    batch_train = batchGenerator(train[:,0:-1],train[:,-1],args.batch_size)
    args.max_batches_train = batch_train.createBatches()
    train(args,batch_train)

if __name__ == "__main__":
    main()