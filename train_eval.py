import time

import tensorflow as tf

from models import *
from read_data import *
from layers import *
from statistics import *
from utils import *

def print_tensors_in_checkpoint_file(file_name):
    """Prints tensors in a checkpoint file.

    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.

    If `tensor_name` is provided, prints the content of the tensor.

    Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    """
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            lprint("tensor_name: ", key)
            lprint(reader.get_tensor(key))
    except Exception as e:  # pylint: disable=broad-except
        lprint(str(e))
        if "corrupted compressed block contents" in str(e):
          lprint("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")


def run_train_epoch(session, model, data, max_batches,sess_summary):
    '''
    Run the model under given session for max_batches based on args
    '''

    start_time = time.time()
    softmax_op = np.zeros((max_batches*model.batch_size,model.op_channels))
    cost_trajectory = []
    y_onehot = np.zeros((max_batches*model.batch_size,model.op_channels))
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

    lprint("Runtime of one epoch: ")
    lprint(end_time-start_time)
    lprint("Average cost per epoch: ")
    lprint(epoch_cost/max_batches)

    return softmax_op, y_onehot, cost_trajectory

def run_val_test_epoch(session, model, data, max_batches,sess_summary):
    '''
    Run the model under given session for max_batches based on args
    '''

    start_time = time.time()
    softmax_op = np.zeros((max_batches*model.batch_size,model.op_channels))
    y_onehot = np.zeros((max_batches*model.batch_size,model.op_channels))

    #print type(model.input_layer_x), type(model.summaries), type(model.output_prob)

    for i in range(max_batches):
        x, y = data.next()
        summary, output_prob = session.run([model.summaries,model.output_prob], feed_dict={model.input_layer_x: x})
        sess_summary.add_summary(summary,i)
        softmax_op[i*len(y):(i+1)*len(y),:] = output_prob
        y_onehot[i*len(y):(i + 1)*len(y),:] = y

    end_time = time.time()

    lprint("Runtime of one epoch: ")
    lprint(end_time-start_time)

    return softmax_op, y_onehot

def train(model_opt, model_args, batch_train, logdir, ix):

    # Initialize session and graph
    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model", reuse=None):

            model_args['mode'] = 'train'
            if model_opt == 'fullDNNNoHistory':
                train_model = fullDNNNoHistory(model_args)
            elif model_opt == 'oneDCNN':
                train_model = oneDCNN(model_args)
            elif model_opt == 'LSTM':
                train_model = LSTM(model_args)
            elif model_opt == 'LSTM_TR':
                train_model = LSTMTargetReplication(model_args)
            else:
                raise ValueError("model specified has not been implemented")

            train_model.build_graph()
            tf.initialize_all_variables().run()

            train_writer = tf.summary.FileWriter(logdir+'model_'+str(ix)+'/train', session.graph)
            saver = tf.train.Saver()
            cost_over_batches = []

            for i in range(model_args['num_epochs']):
                lr_decay = model_args['lr_decay'] ** max(i - 2.0, 0.0)
                train_model.assign_lr(session, model_args['lr_rate'] * lr_decay)

                # run a complete epoch and return appropriate variables
                y_prob, y_onehot, y_cost = run_train_epoch(session, train_model, batch_train, model_args['max_batches_train'], train_writer)

                lprint("For model_"+str(ix)+" Confusion metrics post epoch "+str(i)+" :")
                lprint(compConfusion(y_prob,y_onehot))

                cost_over_batches += y_cost

                if i == model_args['num_epoch']/2:
                    saver.save(session, logdir+'model_'+str(ix)+'/train/train-model-iter', global_step=i)

            saver.save(session, logdir+'model_'+str(ix)+'/train/final-model')
            train_writer.close()

            plt.plot(np.linspace(1,len(cost_over_batches),len(cost_over_batches)),cost_over_batches)
            plt.title('Cost per batch over the training run')
            plt.xlabel('# batch')
            plt.ylabel('avg. cost per batch')
            plt.savefig(logdir+'model_'+str(ix)+'_traincost.png')

def val(model_opt, model_args, batch_val, mode, logdir, ix):
    # Pass mode as either "Val" or "Test"

    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model",reuse=None):

            model_args['mode'] = mode
            if model_opt == 'fullDNNNoHistory':
                val_model = fullDNNNoHistory(model_args)
            elif model_opt == 'oneDCNN':
                val_model = oneDCNN(model_args)
            elif model_opt == 'LSTM':
                val_model = LSTM(model_args)
            elif model_opt == 'LSTM_TR':
                val_model = LSTMTargetReplication(model_args)
            else:
                raise ValueError("model specified has not been implemented")

            val_model.build_graph()
            if model_opt == 'LSTM' or model_opt == 'LSTM_TR':
                val_model.initialize_state(session)

            val_writer = tf.summary.FileWriter(logdir+'model_'+str(ix)+'/val', session.graph)
            restore_var = tf.train.Saver()
            restore_var.restore(session, logdir+'model_'+str(ix)+'/train/final-model')

            #print type(val_model.input_layer_x)
            #print [var.name for var in tf.get_default_graph().get_operations()]
            #print [var.op.name for var in tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

            # run a complete epoch and return appropriate variables
            y_prob, y_onehot = run_val_test_epoch(session, val_model, batch_val, model_args.max_batches_val, val_writer)

            lprint("Confusion metrics post Validation" + model_args['mode'] + " :")
            lprint(compConfusion(y_prob, y_onehot))

            rocPrAuc(y_prob, y_onehot, logdir, 'val')

            val_writer.close()

