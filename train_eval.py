import time

import tensorflow as tf

from models import *
from read_data import *
from layers import *
from statistics import *


def run_train_epoch(session, model, data, args, max_batches,sess_summary):
  '''
  Run the model under given session for max_batches based on args
  '''

  start_time = time.time()
  row, col = data.get_data_size()
  softmax_op = np.zeros((row,args.op_channels))
  cost_trajectory = []
  y_onehot = np.zeros((row,args.op_channels))
  epoch_cost = 0.0

  for i in range(max_batches):
    x, y = data.next()
    summary, cur_cost, output_prob, _ = session.run([model.summaries,model.cost,model.output_prob,model.train_op],
                feed_dict={model.input_layer_x: x, model.input_layer_y: y})
    sess_summary.add_summary(summary,i)
    cost_trajectory.append(cur_cost)
    softmax_op[i*len(y):(i+1)*len(y),:] = output_prob
    y_onehot[i*len(y):(i+1)*len(y),:] = y
    epoch_cost += cur_cost

    #print "Batch Cross Entropy Loss: "
    #print cur_cost

  end_time = time.time()

  print "Runtime of one epoch: "
  print end_time-start_time
  print "Average cost per epoch: "
  print epoch_cost/max_batches

  return softmax_op, y_onehot, cost_trajectory

def run_val_test_epoch(session, model, data, args, max_batches,sess_summary):
  '''
  Run the model under given session for max_batches based on args
  '''

  start_time = time.time()
  row, col = data.get_data_size()
  softmax_op = np.zeros((row,args.op_channels))
  y_onehot = np.zeros((row,args.op_channels))

  for i in range(max_batches):
    x, y = data.next()
    print type(x)
    summary, output_prob = session.run([model.summaries,model.output_prob],
                feed_dict={model.input_layer_x: x})
    sess_summary.add_summary(summary,i)
    softmax_op[i*len(y):(i+1)*len(y),:] = output_prob
    y_onehot[i*len(y):(i + 1)*len(y),:] = y

  end_time = time.time()

  print "Runtime of one epoch: "
  print end_time-start_time

  return softmax_op, y_onehot

def train(args,batch_train):

    # Initialize session and graph
    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model", reuse=None):

            args.mode = 'train'
            if args.model == 'fullDNNNoHistory':
                train_model = fullDNNNoHistory(args)
            elif args.model == 'oneDCNN':
                train_model = oneDCNN(args)
            elif args.model == 'LSTM':
                train_model = LSTM(args)
            else:
                raise ValueError("model specified has not been implemented")

            train_model.build_graph()
            tf.initialize_all_variables().run()

            train_writer = tf.train.SummaryWriter(args.logdir+'/train',session.graph)
            saver = tf.train.Saver()
            cost_over_batches = []

            for i in range(args.num_epochs):
                lr_decay = args.lr_decay ** max(i - 2.0, 0.0)
                train_model.assign_lr(session, args.lr_rate * lr_decay)

                # run a complete epoch and return appropriate variables
                y_prob, y_onehot, y_cost = run_train_epoch(session, train_model, batch_train, args, args.max_batches_train,train_writer)

                print "Confusion metrics post epoch "+str(i)+" :"
                print compConfusion(y_prob,y_onehot)

                cost_over_batches += y_cost

                if i%5 ==0:
                    saver.save(session,args.logdir+'/train/train-model-iter',global_step=i)

            saver.save(session,args.logdir+'/train/final-model')
            train_writer.close()

            plt.plot(np.linspace(1,len(cost_over_batches),len(cost_over_batches)),cost_over_batches)
            plt.title('Cost per batch over the training run')
            plt.xlabel('# batch')
            plt.ylabel('avg. cost per batch')
            plt.show()

def val(args,batch_val,mode):
    # Pass mode as either "Val" or "Test"

    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model"):

            args.mode = 'val'
            if args.model == 'fullDNNNoHistory':
                val_model = fullDNNNoHistory(args)
            elif args.model == 'oneDCNN':
                val_model = oneDCNN(args)
            elif args.model == 'LSTM':
                val_model = LSTM(args)
            else:
                raise ValueError("model specified has not been implemented")

            val_model.build_graph()
            val_writer = tf.train.SummaryWriter(args.logdir+'/val',session.graph)
            restore_var = tf.train.Saver()
            restore_var.restore(session, args.logdir+'/train/final-model')

            # run a complete epoch and return appropriate variables
            y_prob, y_onehot = run_val_test_epoch(session, val_model, batch_val, args, args.max_batches_train,val_writer)

            print "Confusion metrics post Validation"+args.mode+" :"
            print compConfusion(y_prob, y_onehot)

            val_writer.close()