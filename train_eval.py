import time

import tensorflow as tf

from models import *
from read_data import *
from layers import *
from statistics import *
import logging

module_logger = logging.getLogger('timeSeriesDL.train_eval')


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
            module_logger.info("tensor_name: ", key)
            module_logger.info(reader.get_tensor(key))
    except Exception as e:  # pylint: disable=broad-except
        module_logger.info(str(e))
        if "corrupted compressed block contents" in str(e):
            module_logger.info("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

class EarlyStopping(object):
    '''
    A basic implementation of early stopping based on Keras Callbacks. Saves the best model for
    reference and updates it as required (missing in Keras EarlyStoppin).
    '''

    def __init__(self,monitorargs):

        for key in ['monitor','min_delta','patience','mode']:
            if key not in monitorargs.keys():
                raise ValueError(key+" not in provided dictionary")

        self.monitor = monitorargs['monitor']
        self.min_delta = monitorargs['min_delta']
        self.patience = monitorargs['patience']
        self.mode = monitorargs['mode']
        self.best = np.Inf if self.monitor == np.less else -np.Inf
        self.monitor_op = np.greater if self.mode == 'max' else np.less
        self.wait = 0

    def check_model(self,curr_monitor_val):

        if self.monitor_op(curr_monitor_val - self.min_delta,self.best):
            self.best = curr_monitor_val
            self.wait = 0
            return 'save_model'
        else:
            if self.wait >= self.patience:
                return 'terminate_model'

            self.wait += 1
            return 'continue'

def run_train_epoch(session, model, data, max_batches,sess_summary, epoch_num):
    '''
    Run the model under given session for max_batches based on args
    '''

    softmax_op = np.zeros((max_batches*model.batch_size,model.op_channels))
    cost_trajectory = []
    y_onehot = np.zeros((max_batches*model.batch_size,model.op_channels))
    epoch_cost = 0.0

    for i in range(max_batches):
        x, y = data.next()
        summary, cur_cost, output_prob, _ = session.run([model.summaries,model.cost,model.output_prob,model.train_op],
                    feed_dict={model.input_layer_x: x, model.input_layer_y: y})
        if sess_summary:
            sess_summary.add_summary(summary,i)
        cost_trajectory.append(cur_cost)
        softmax_op[i*len(y):(i+1)*len(y),:] = output_prob
        y_onehot[i*len(y):(i+1)*len(y),:] = y
        epoch_cost += cur_cost

    avg_cost = epoch_cost/max_batches
    module_logger.info("Epcoh: %d, Average cost per epoch: %f",epoch_num,avg_cost)

    return softmax_op, y_onehot, cost_trajectory

def run_val_test_epoch(session, model, data, max_batches,sess_summary):
    '''
    Run the model under given session for max_batches based on args
    '''

    softmax_op = np.zeros((max_batches*model.batch_size,model.op_channels))
    y_onehot = np.zeros((max_batches*model.batch_size,model.op_channels))

    for i in range(max_batches):
        x, y = data.next()
        summary, output_prob = session.run([model.summaries,model.output_prob], feed_dict={model.input_layer_x: x,
                                                                                           model.input_layer_y: y})
        if sess_summary:
            sess_summary.add_summary(summary,i)
        softmax_op[i*len(y):(i+1)*len(y),:] = output_prob
        y_onehot[i*len(y):(i + 1)*len(y),:] = y

    return softmax_op, y_onehot

def train(model_opt, model_args, batch_train, batch_val, logdir, ix, early_stop_args, summ_flag):

    # setup the early stopping routine with parameters passed
    valmonitor = EarlyStopping(early_stop_args)

    # Initialize session and graph
    # Limit usage to 50% of the GPU resources available
    gconfig = tf.ConfigProto()
    gconfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Graph().as_default(), tf.Session(config=gconfig) as session:

        if model_opt == 'cnn':
            model = oneDCNN
        elif model_opt == 'lstm':
            model = LSTM
        elif model_opt == 'lstm_tr':
            model = LSTMTargetReplication
        else:
            raise ValueError("model specified has not been implemented")

        with tf.variable_scope("model", reuse=None):
            model_args['mode'] = 'train'
            train_model = model(model_args)
            train_model.build_graph()

            tf.global_variables_initializer().run()

            train_writer = None
            val_writer = None
            if summ_flag:
                train_writer = tf.summary.FileWriter(logdir+'/model_'+str(ix)+'/train', session.graph)
                val_writer = tf.summary.FileWriter(logdir + '/model_' + str(ix) + '/val', session.graph)
            saver = tf.train.Saver()
            cost_over_batches = []

            best_y_val_prob, best_y_val_onehot = None, None
            for i in range(model_args['num_epochs']):

                lr_decay = model_args['lr_decay'] ** max(i - 2.0, 0.0)
                train_model.assign_lr(session, model_args['lr_rate'] * lr_decay)

                # run a complete epoch and return appropriate variables
                y_prob, y_onehot, y_cost = run_train_epoch(session, train_model, batch_train, model_args['max_batches_train'], train_writer, i)
                cost_over_batches += y_cost
                print(len(cost_over_batches))

                # For every val_check epoch, check the validation scores and see if they have improved from the previous best
                y_val_prob, y_val_onehot = run_val_test_epoch(session, train_model, batch_val, model_args['max_batches_val'], val_writer)
                curr_pr, curr_roc, curr_f1, curr_acc = compMetrics(y_val_prob,y_val_onehot)

                if early_stop_args['monitor']  == 'pr':
                    opt = valmonitor.check_model(curr_pr)
                elif early_stop_args['monitor'] == 'roc':
                    opt = valmonitor.check_model(curr_roc)
                elif early_stop_args['monitor'] == 'f1':
                    opt = valmonitor.check_model(curr_f1)
                elif early_stop_args['monitor'] == 'acc':
                    opt = valmonitor.check_model(curr_acc)
                else:
                    raise ValueError("early stopping monitor not in list")

                if opt == 'save_model':
                    saver.save(session, logdir + '/model_' + str(ix) + '/train/best-model')
                    best_y_val_prob = y_val_prob
                    best_y_val_onehot = y_val_onehot
                if opt == 'terminate_model':
                    break


            if summ_flag:
                train_writer.close()

            costTrainPlot(cost_over_batches, logdir, '/model_' + str(ix))
            roc_auc, pr_auc = rocPrAucPlot(best_y_val_prob, best_y_val_onehot, logdir, '/model_' + str(ix))
            return roc_auc, pr_auc

def test(model_opt, model_args, batch_val, mode, logdir, ix, summ_flag):
    # Pass mode as either "Val" or "Test"

    # Limit usage to 50% of the GPU resources available
    gconfig = tf.ConfigProto()
    gconfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Graph().as_default(), tf.Session(config=gconfig) as session:

        with tf.variable_scope("model",reuse=None):

            model_args['mode'] = mode
            if model_opt == 'cnn':
                val_model = oneDCNN(model_args)
            elif model_opt == 'lstm':
                val_model = LSTM(model_args)
            elif model_opt == 'lstm_tr':
                val_model = LSTMTargetReplication(model_args)
            else:
                raise ValueError("model specified has not been implemented")

            val_model.build_graph()
            if model_opt == 'lstm' or model_opt == 'lstm_tr':
                val_model.initialize_state(session)

            val_writer = None
            if summ_flag:
                val_writer = tf.summary.FileWriter(logdir+'/model_'+str(ix)+'/val', session.graph)
            restore_var = tf.train.Saver()
            restore_var.restore(session, logdir+'/model_'+str(ix)+'/train/best-model')

            # run a complete epoch and return appropriate variables
            y_prob, y_onehot = run_val_test_epoch(session, val_model, batch_val, model_args['max_batches_val'], val_writer)

            module_logger.info("Confusion metrics post Validation" + model_args['mode'] + " :")
            module_logger.info(compConfusion(y_prob, y_onehot))

            roc_auc, pr_auc = rocPrAucPlot(y_prob, y_onehot, logdir, '/model_'+str(ix))

            if summ_flag:
                val_writer.close()

            return roc_auc, pr_auc

