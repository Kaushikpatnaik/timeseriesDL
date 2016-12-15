import sys
import time

import tensorflow as tf
import argparse

import numpy as np
from models import *
from read_data import *
from layers import *
import cPickle

def run_epoch(session, model, data, args, max_batches,sess_summary):
  '''
  Run the model under given session for max_batches based on args
  '''

  tot_cost = 0.0
  start_time = time.time()
  iters = 0

  for i in range(max_batches):
    x, y = data.next()
    summary, cur_cost, output, _ = session.run([model.summaries,model.cost,model.output,model.train_op],
                feed_dict={model.input_layer_x: x, model.input_layer_y: y})
    sess_summary.add_summary(summary,i)
    print "Batch Cross Entropy Loss: "
    print float(cur_cost)/args.batch_size
    iters += args.batch_size

  end_time = time.time()

  print "Runtime of one epoch: "
  print end_time-start_time

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

            train_model.build_graph()
            tf.initialize_all_variables().run()

            train_writer = tf.train.SummaryWriter(args.logdir+'/train',session.graph)

            for i in range(args.num_epochs):
                # TODO: Add parameter for max_max_epochs
                lr_decay = args.lr_decay ** max(i - 2.0, 0.0)
                train_model.assign_lr(session, args.lr_rate * lr_decay)

                # run a complete epoch and return appropriate variables
                run_epoch(session, train_model, batch_train, args, args.max_batches_train,train_writer)

            train_writer.close()

def val(args,batch_val,train_model):

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1,0.1)

        with tf.variable_scope("model", reuse=True, initializer=initializer):

            args.mode = "Val"
            if args.model_opt == 'fullDNNNoHistory':
                val_model = fullDNNNoHistory(args)
            elif args.model_opt == 'fullDNNWithHistory':
                val_model = fullDNNWithHistory(args)
            else:
                raise NotImplementedError

            val_model.build_graph()
            tf.initialize_all_variables().run()

            val_writer = tf.train.SummaryWriter(args.logdir+'/val',session.graph)

            for i in range(args.num_epochs):

                # run a complete epoch and return appropriate variables
                run_epoch(session, train_model, batch_train, args, args.max_batches_train,train_writer)

            train_writer.close()



def test(args):

    raise NotImplementedError

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirloc', type=str, default='./data/backblaze',
                        help='data location for all data')
    parser.add_argument('--split_ratio', type=list, default=[0.8, 0.1, 0.1],
                        help='split ratio for train, validation and test')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data')
    parser.add_argument('--layer_sizes', type=list, default=[32,16], help='number of hidden units in the cell')
    parser.add_argument('--op_channels', type= int, default=7, help='number of output classes')
    parser.add_argument('--num_epochs', type=int, default=5, help='max number of epochs to run the training')
    parser.add_argument('--lr_rate', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--drive_model', type=str, default='ST3000DM001', help='drive model for model building')
    parser.add_argument('--hist', type=int, default=5, help='history to use when predicting')
    parser.add_argument('--pred_window', type=int, default=3, help='lookahead to be used for prediction')
    parser.add_argument('--model_opt', type=str, default='fullDNNNoHistory',help='model to be tested')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')


    args = parser.parse_args()
    #backblaze_data = blackblazeReader(args)
    #train, val, test = backblaze_data.train_test_split(args.split_ratio)

    # Saved the train, val and test sets for future work, as they take a lot of time to prepare
    #cPickle.dump(train, open('./data/backblaze_' + str(args.drive_model) + '_train.pkl','w'))
    #cPickle.dump(val, open('./data/backblaze_' + str(args.drive_model) + '_val.pkl', 'w'))
    #cPickle.dump(test, open('./data/backblaze_' + str(args.drive_model) + '_test.pkl', 'w'))

    #train_data = cPickle.load(open('./data/backblaze_' + str(args.drive_model) + '_train.pkl','rb'))
    #val_data = cPickle.load(open('./data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))
    #test_data = cPickle.load(open('./data/backblaze_' + str(args.drive_model) + '_train.pkl', 'rb'))

    train_data_raw = open('./data/ElectricDevices_TRAIN','r+')
    ucr_data = ucrDataReader(train_data_raw,0.8,args.op_channels)
    train_data, val_data = ucr_data.trainTestSplit()

    # Training section
    print "Training Dataset Shape: "
    print train_data.shape

    batch_train = batchGenerator(train_data,args.batch_size,args.op_channels)
    batch_train.createBatches()
    args.max_batches_train = batch_train.get_num_batches()
    args.ip_channels = batch_train.get_ip_channels()
    train(args,batch_train)

    # Validation section
    print "Validation DataSet Shape: "
    print val_data.shape

    batch_val = batchGenerator(val_data,1,args.op_channels)
    batch_val.createBatches()
    args.max_batches_train = batch_val.get_num_batches()
    args.ip_channels = batch_val.get_ip_channels()
    val(args,batch_val)

if __name__ == "__main__":
    main()