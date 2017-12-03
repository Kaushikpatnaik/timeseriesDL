import tensorflow as tf
import argparse


from read_data import *
from train_eval import *
import time
from collections import OrderedDict, defaultdict
from itertools import product
from sklearn.grid_search import ParameterGrid
import logging
import os

def main():

    today = time.strftime("%d_%m_%Y %H_%M_%S")
    logdir = './logs/'+today
    logfile = 'logfile'
    summary_flag = False
    early_stop_args = {'monitor': 'f1', 'min_delta': 0.002, 'patience': 10, 'mode': 'max'}

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)-8s %(message)s', \
                        handlers=[logging.FileHandler("{0}/{1}.log".format(logdir,logfile)),logging.StreamHandler()])
    logger = logging.getLogger('timeSeriesDL')

    f = open(logdir + '/results.csv', 'w+')

    for itr_data, itr_model in product(data_list,model_list):

        logger.info("Running model: "+str(itr_model)+" for dataset: "+str(itr_data))

        # grid of params for the data and the model
        args_data = ParameterGrid(data_args[itr_data])
        args_model = ParameterGrid(model_args[itr_model])

        # Each model may have a grid of parameters to check over
        # There may be data parameters for each dataset also !
        # ideally the data for such a grid check should be consistent in terms of train, val and test split
        for it, data_param in enumerate(args_data):

            # based on the data_param determine the dataset to be downloaded and split
            train_data, val_data, test_data, ip_channel, op_channel, seq_len = get_data_obj(data_param,itr_data)

            if not os.path.exists(logdir):
                os.makedirs(logdir)

            # TODO: Kaushik use a json string to write data params, model params and results

            for ix, model_param in enumerate(args_model):
                f.write(str(data_param.items())+'\t')
                f.write(str(it)+'_'+str(ix)+'\t'+str(model_param.items())+'\t')

                if not os.path.exists(logdir+'/model_'+str(ix)):
                    os.makedirs(logdir+'/model_'+str(ix))

                logger.info("Training Dataset Shape: ")
                logger.info(train_data.shape)
                batch_train = balBatchGenerator(train_data, data_param['batch_size'], ip_channel,
                                                op_channel, seq_len, data_param['label_ratio'])

                logger.info("Validation DataSet Shape: ")
                logger.info(val_data.shape)
                batch_val = batchGenerator(val_data, data_param['batch_size'], ip_channel, op_channel, seq_len)

                model_param['max_batches_train'] = data_param['max_batches_train']
                model_param['ip_channels'] = ip_channel
                model_param['op_channels'] = op_channel
                model_param['seq_len'] = seq_len
                model_param['batch_size'] = data_param['batch_size']
                model_param['weights'] = data_param['label_weights']
                model_param['max_batches_val'] = batch_val.get_num_batches()

                # train with early stopping and return best model's pr and roc auc
                roc_auc, pr_auc = train(itr_model, model_param, batch_train, batch_val, logdir, ix, early_stop_args, summary_flag)
                f.write(str(roc_auc) + '\t' + str(pr_auc))
                f.write('\n')

                # test performance
                #roc_auc, pr_auc = test(itr_model, model_param, batch_val, "val", logdir, ix, summary_flag)

    f.close()


if __name__ == "__main__":
    dnn_args = {'layer_sizes': [[64, 32], [64, 32, 16], [128, 64, 32]], 'num_epochs': [50],
                'lr_rate': [0.001, 0.0001], 'lr_decay': [0.9], 'weight_reg': [0.01, 0.001],
                'cost_reg': [0.01, 0.001]}

    lstm_args = {'cell': 'lstm', 'num_layers': [1, 2, 3], 'hidden_units': [16, 32, 64],
                 'num_epochs': [50], 'lr_rate': [0.001, 0.0001], 'lr_decay': 0.97,
                 'grad_clip': [3.0, 5.0]}

    lstmtr_args = {'cell': ['lstm'], 'num_layers': [1, 2, 3], 'hidden_units': [16, 32, 64],
                   'num_epochs': [50], 'lr_rate': [0.001, 0.0001], 'lr_decay': 0.97,
                   'grad_clip': [3.0, 5.0]}

    # TODO: Add more layers
    layer_params_opt_1 = OrderedDict({'2_full': (64), '1_conv': (3, 64, 1, 'VALID')})
    layer_params_opt_2 = OrderedDict({'2_full': (256), '1_conv': (3, 128, 1, 'VALID')})
    layer_params_opt_3 = OrderedDict({'3_full': (32), '2_conv': (3, 32, 1, 'VALID'), '1_conv': (1, 64, 1, 'VALID')})
    cnn_args = {'num_epochs': [30], 'lr_rate': [0.001, 0.0001],
                'lr_decay': [0.9], 'layer_params': [layer_params_opt_3], 'weight_reg': [0.01, 0.001],
                'cost_reg': [0.01, 0.001]}

    '''
    pred_window here controls the number of days assigned as positive/failure
    hist controls the number of days being used for prediction
    '''
    list_label_ratio = [{0: 0.9, 1: 0.1}, {0: 0.7, 1: 0.3}, {0: 0.5, 1: 0.5}]
    list_label_weight = [[0.2, 1], [0.5, 1]]
    data_args = {
        'backblaze': {'batch_size': [64], 'split_ratio': [[0.8, 0.1, 0.1]], 'label_ratio': list_label_ratio,
                      'label_weights': list_label_weight, 'drive_model': ['ST3000DM001','ST4000DM000'], \
                      'max_batches_train': [5000], 'hist': [4,7], 'pred_window': [3], 'year': ['2015']}}

    model_args = {'lstm': lstm_args, 'cnn': cnn_args, 'lstrm_lr': lstmtr_args, 'dnn': dnn_args}

    # replacing the data and model arguments with dictionaries to run multiple experiments
    data_list = ['backblaze']
    model_list = ['cnn','lstm','lstm_tr']


    main()