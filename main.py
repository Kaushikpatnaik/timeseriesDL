import tensorflow as tf
import argparse

import numpy as np
from read_data import *
from train_eval import *
import json
from collections import OrderedDict, defaultdict
from utils import *

dnn_args = {'layer_sizes': [[64,32],[64,32,16],[128,64,32]], 'num_epochs': [5,7], 'lr_rate': [0.01,0.001,0.0001], 'lr_decay': [0.97,0.9,0.8]}

lstm_args = {'cell': ['lstm'], 'num_layers': [1,2,3], 'hidden_units': [16,32,64,128], 'num_epochs': [8,10,12], 'lr_rate': [0.001,0.0001], 'lr_decay': [0.97,0.9], 'grad_clip': [3.0,5.0,8.0]}

lstmtr_args = {'cell': ['lstm'], 'num_layers': [1,2,3], 'hidden_units': [16,32,64,128], 'num_epochs': [8,10,12], 'lr_rate': [0.001,0.0001], 'lr_decay': [0.97,0.9], 'grad_clip': [3.0,5.0,8.0]}

# TODO: Add more layers
layer_params_2 = OrderedDict({'2_full': (64), '1_conv': (3,64,1,'VALID')})
cnn_args = {'num_layers': [2,3,4], 'num_epochs': [5,8,10], 'lr_rate': [0.01,0.001,0.0001], 'lr_decay': [0.97,0.9], 'layer_params': [layer_params_2,]}

# TODO: Make the num_layers redundant or derivative from layer_params
cnn_multi_args = {'num_layers': [2,3,4], 'num_epochs': [5,8,10], 'lr_rate': [0.01,0.001,0.0001], 'lr_decay': [0.97,0.9], 'layer_params': [layer_params_2,], 'sub_sample': []}

list_label_ratio = [{},{},{},{}]
list_label_weight = []
data_args = {'backblaze': {'batch_size': [64,128,256], 'split_ratio': [[0.8,0.1,0.1]], 'label_ratio': list_label_ratio}, 'electric': {'batch_size': [64], 'split_ratio': [[0.8,0.1,0.1]]}}

model_args = {'lstm': lstm_args, 'cnn': cnn_args, 'lstrm_lr': lstmtr_args, 'dnn': dnn_args, 'multi_cnn': cnn_multi_args}

def main():

    # replacing the data and model arguments with dictionaries to run multiple experiments
    data_list = ['blackblaze','electric']
    model_list = ['cnn','multi_cnn','lstm','lstm_tr','dnn']

    for itr_data, itr_model in product(data_list,model_list):

        print "Running model: "+str(itr_model)+" for dataset: "+str(itr_data)

        # grid of params for the data and the model
        args_data = param(data_args[itr_data])
        args_model = param(model_args[itr_model])

        # Each model may have a grid of parameters to check over
        # There may be data parameters for each dataset also !
        # ideally the data for such a grid check should be consistent in terms of train, val and test split

        for data_param in args_data:

            # based on the data_param determine the dataset to be downloaded and split
            train_data, val_data, test_data, ip_channel, op_channel, seq_len = get_data_obj(data_param)

            for model_param in args_model:




                # Training section
                print "Training Dataset Shape: "
                print train_data.shape

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

                label_ratio = param['label_ratio']
                batch_train = balBatchGenerator(train_data_new, args_data.batch_size, args_data.ip_channels,
                                                args_data.op_channels, args_data.seq_len, label_ratio)

                batch_val = batchGenerator(val_data_new, 64, args_data.ip_channels, args_data.op_channels, args_data.seq_len)

                args_model.max_batches_train = args_data['max_batches_train']
                args_model.ip_channels = args_data.ip_channels
                args_model.op_channels = args_data.op_channels
                args_model.seq_len = args_data.seq_len
                args_model.batch_size = args_data.batch_size
                args_model.weights = [1.0/5,1.0]

                # train and return the saved trainable parameters of the model
                train(args_model,batch_train)

                # Validation section
                print "Validation DataSet Shape: "
                print val_data.shape


                args_model.max_batches_val = batch_val.get_num_batches()
                args_model.batch_size = 64

                val(args_model,batch_val,"val")


if __name__ == "__main__":
    main()