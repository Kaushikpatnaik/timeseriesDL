import tensorflow as tf
import argparse


from read_data import *
from train_eval import *
import time
from collections import OrderedDict, defaultdict
from utils import *
import logging
import os

def main():

    # replacing the data and model arguments with dictionaries to run multiple experiments
    data_list = ['backblaze','electric']
    model_list = ['cnn','lstm','lstm_tr']

    today = time.strftime("%d_%m_%Y %H_%M_%S")
    logdir = './logs/'+today+'/'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.basicConfig(level=logging.DEBUG, filename=logdir+'logfile', filemode='a+', \
                        format='%(asctime)-15s %(levelname)-8s %(message)s')

    for itr_data, itr_model in product(data_list,model_list):

        lprint("Running model: "+str(itr_model)+" for dataset: "+str(itr_data))

        # grid of params for the data and the model
        args_data = paramGrid(data_args[itr_data])
        args_model = paramGrid(model_args[itr_model])

        # Each model may have a grid of parameters to check over
        # There may be data parameters for each dataset also !
        # ideally the data for such a grid check should be consistent in terms of train, val and test split
        for it, data_param in enumerate(args_data):

            # based on the data_param determine the dataset to be downloaded and split
            train_data, val_data, test_data, ip_channel, op_channel, seq_len = get_data_obj(data_param,itr_data)

            logdir += 'data_param_'+str(it)+'/'
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            f = open(logdir+'config.csv','w+')
            f.write(str(it)+'\t'+str(data_param.items()))

            for ix, model_param in enumerate(args_model):

                f.write(str(it)+'_'+str(ix)+'\t'+str(model_param.items()))

                # Training section
                lprint("Training Dataset Shape: ")
                lprint(train_data.shape)

                '''
                if args_model.model == 'oneDMultiChannelCNN':
                    train_data_new, args_model.sub_sample_lens = low_pass_and_subsample(train_data)
                    val_data_new, args_model.sub_sample_lens = low_pass_and_subsample(val_data)
                    test_data_new, args_model.sub_sample_lens = low_pass_and_subsample(test_data)
                elif args_model.model == 'freqCNN':
                    train_data_new = freq_transform(train_data)
                    val_data_new = freq_transform(val_data)
                    test_data_new = freq_transform(test_data)
                else:
                    train_data_new = train_data
                    val_data_new = val_data
                    test_data_new = test_data
                '''

                batch_train = balBatchGenerator(train_data, data_param['batch_size'], ip_channel,
                                                op_channel, seq_len, data_param['label_ratio'])

                batch_val = batchGenerator(val_data, data_param['batch_size'], ip_channel, op_channel, seq_len)

                model_param['max_batches_train'] = data_param['max_batches_train']
                model_param['ip_channels'] = ip_channel
                model_param['op_channels'] = op_channel
                model_param['seq_len'] = seq_len
                model_param['batch_size'] = data_param['batch_size']
                model_param['weights'] = data_param['label_weights']

                # train and return the saved trainable parameters of the model
                train(itr_model, model_param, batch_train, logdir, ix)

                # Validation section
                lprint("Validation DataSet Shape: ")
                lprint(val_data.shape)

                model_param['max_batches_val'] = batch_val.get_num_batches()
                val(itr_model, model_param, batch_val, "val", logdir, ix)

            f.close()


if __name__ == "__main__":
    dnn_args = {'layer_sizes': [[64, 32], [64, 32, 16], [128, 64, 32]], 'num_epochs': [5, 7],
                'lr_rate': [0.01, 0.001, 0.0001], 'lr_decay': [0.97, 0.9, 0.8], 'weight_reg': [1, 0.1, 0.01, 0.001],
                'cost_reg': [1, 0.1, 0.01, 0.001]}

    lstm_args = {'cell': ['lstm'], 'num_layers': [1, 2, 3], 'hidden_units': [16, 32, 64, 128],
                 'num_epochs': [8, 10, 12], 'lr_rate': [0.001, 0.0001], 'lr_decay': [0.97, 0.9],
                 'grad_clip': [3.0, 5.0, 8.0]}

    lstmtr_args = {'cell': ['lstm'], 'num_layers': [1, 2, 3], 'hidden_units': [16, 32, 64, 128],
                   'num_epochs': [8, 10, 12], 'lr_rate': [0.001, 0.0001], 'lr_decay': [0.97, 0.9],
                   'grad_clip': [3.0, 5.0, 8.0]}

    # TODO: Add more layers
    layer_params_opt_1 = OrderedDict({'2_full': (64), '1_conv': (3, 64, 1, 'VALID')})
    layer_params_opt_2 = OrderedDict({'2_full': (256), '1_conv': (3, 128, 1, 'VALID')})
    layer_params_opt_3 = OrderedDict({'3_full': (64), '2_conv': (3, 32, 1, 'VALID'), '1_conv': (3, 64, 1, 'VALID')})
    cnn_args = {'num_epochs': [5, 8, 10], 'lr_rate': [0.01, 0.001, 0.0001],
                'lr_decay': [0.97, 0.9], 'layer_params': [layer_params_opt_1, layer_params_opt_2, layer_params_opt_3], 'weight_reg': [1.0, 0.1, 0.01, 0.001],
                'cost_reg': [1.0, 0.1, 0.01, 0.001]}

    # TODO: Make the num_layers redundant or derivative from layer_params
    cnn_multi_args = {'num_layers': [2, 3, 4], 'num_epochs': [5, 8, 10], 'lr_rate': [0.01, 0.001, 0.0001],
                      'lr_decay': [0.97, 0.9], 'layer_params': [layer_params_opt_1, layer_params_opt_2, layer_params_opt_3], 'sub_sample': [],
                      'weight_reg': [1, 0.1, 0.01, 0.001], 'cost_reg': [1, 0.1, 0.01, 0.001]}

    list_label_ratio = [{0: 0.9, 1: 0.1}, {0: 0.7, 1: 0.3}, {0: 0.5, 1: 0.5}]
    list_label_weight = [[0.2, 1], [0.5, 1], [1, 1]]
    data_args = {
        'backblaze': {'batch_size': [64, 128, 256], 'split_ratio': [[0.8, 0.1, 0.1]], 'label_ratio': list_label_ratio,
                      'label_weights': list_label_weight, 'drive_model': ['ST3000DM001','ST4000DM000'], 'max_batches_train': [5000]}, \
        'electric': {'batch_size': [64], 'split_ratio': [[0.8, 0.1, 0.1]], 'label_ratio': list_label_ratio,
                     'label_weights': list_label_weight}}

    model_args = {'lstm': lstm_args, 'cnn': cnn_args, 'lstrm_lr': lstmtr_args, 'dnn': dnn_args,
                  'multi_cnn': cnn_multi_args}



    main()