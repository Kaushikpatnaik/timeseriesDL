import sys
import time

import tensorflow as tf
import argparse

import numpy as np
from models import *
from read_data import *
from layers import *
from statistics import *

def run_train_epoch(session, model, data, args, max_batches,sess_summary):
  '''
  Run the model under given session for max_batches based on args
  '''

  start_time = time.time()
  softmax_op = np.zeros((max_batches*args.batch_size,args.op_channels))
  cost_trajectory = []
  y_onehot = np.zeros((max_batches*args.batch_size,args.op_channels))

  for i in range(max_batches):
    x, y = data.next()
    summary, cur_cost, output_prob, _ = session.run([model.summaries,model.cost,model.output_prob,model.train_op],
                feed_dict={model.input_layer_x: x, model.input_layer_y: y})
    sess_summary.add_summary(summary,i)
    cost_trajectory.append(cur_cost)
    softmax_op[i*args.batch_size:(i+1)*args.batch_size,:] = output_prob
    y_onehot[i*args.batch_size:(i+1)*args.batch_size,:] = y

    print "Batch Cross Entropy Loss: "
    print cur_cost

  end_time = time.time()

  print "Runtime of one epoch: "
  print end_time-start_time

  return softmax_op, y_onehot, cost_trajectory

def run_val_test_epoch(session, model, data, args, max_batches,sess_summary):
  '''
  Run the model under given session for max_batches based on args
  '''

  start_time = time.time()
  softmax_op = np.zeros((max_batches*args.batch_size,args.op_channels))
  y_onehot = np.zeros((max_batches*args.batch_size,args.op_channels))

  for i in range(max_batches):
    x, y = data.next()
    summary, output_prob = session.run([model.summaries,model.output_prob],
                feed_dict={model.input_layer_x: x})
    sess_summary.add_summary(summary,i)
    softmax_op[i*args.batch_size:(i+1)*args.batch_size,:] = output_prob
    y_onehot[i * args.batch_size:(i + 1) * args.batch_size, :] = y

  end_time = time.time()

  print "Runtime of one epoch: "
  print end_time-start_time

  return softmax_op, y_onehot

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
            saver = tf.train.Saver()
            cost_over_batches = []

            for i in range(args.num_epochs):
                # TODO: Add parameter for max_max_epochs
                lr_decay = args.lr_decay ** max(i - 2.0, 0.0)
                train_model.assign_lr(session, args.lr_rate * lr_decay)

                # run a complete epoch and return appropriate variables
                y_prob, y_onehot, y_cost = run_train_epoch(session, train_model, batch_train, args, args.max_batches_train,train_writer)

                print "Confusion metrics post epoch "+str(i)+" :"
                print compConfusion(y_prob,y_onehot)

                cost_over_batches += y_cost

                if i%5 ==0:
                    saver.save(session,'./logs/train/train-model-iter',global_step=i)

            saver.save(session,'./logs/train/final-model')
            train_writer.close()

            print "Len cost over batches: "
            print len(cost_over_batches)

            plt.plot(np.linspace(1,len(cost_over_batches),len(cost_over_batches)),cost_over_batches)
            plt.title('Cost per batch over the training run')
            plt.xlabel('# batch')
            plt.ylabel('avg. cost per batch')
            plt.show()

def val(args,batch_val,mode):
    # Pass mode as either "Val" or "Test"

    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model"):

            args.mode = mode
            if args.model_opt == 'fullDNNNoHistory':
                val_model = fullDNNNoHistory(args)
            elif args.model_opt == 'fullDNNWithHistory':
                val_model = fullDNNWithHistory(args)
            else:
                raise NotImplementedError

            val_model.build_graph()
            val_writer = tf.train.SummaryWriter(args.logdir+'/val',session.graph)
            restore_var = tf.train.Saver()
            restore_var.restore(session, './logs/train/final-model')

            # run a complete epoch and return appropriate variables
            y_prob, y_onehot = run_val_test_epoch(session, val_model, batch_val, args, args.max_batches_train,val_writer)

            print "Confusion metrics post "+args.mode+" :"
            print compConfusion(y_prob, y_onehot)

            val_writer.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirloc', type=str, default='./data/backblaze',
                        help='data location for all data')
    parser.add_argument('--split_ratio', type=list, default=[0.8, 0.1, 0.1],
                        help='split ratio for train, validation and test')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data')
    parser.add_argument('--layer_sizes', type=list, default=[64,32,16], help='number of hidden units in the cell')
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

    # train and return the saved trainable parameters of the model
    train(args,batch_train)

    # Validation section
    print "Validation DataSet Shape: "
    print val_data.shape

    batch_val = batchGenerator(val_data,64,args.op_channels)
    batch_val.createBatches()
    args.max_batches_train = batch_val.get_num_batches()
    args.ip_channels = batch_val.get_ip_channels()
    val(args,batch_val,"val")

if __name__ == "__main__":
    main()