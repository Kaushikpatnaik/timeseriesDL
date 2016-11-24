import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class ucrDataReader(object):
    '''
    UCR time series has only one channel
    '''

    def __init__(self,raw_data,train_test_split,batch_size):

        self.raw_data = raw_data
        self.batch_size = batch_size
        self.train_test_split = train_test_split


    def _preProcess(self):

        data = []
        y = []
        for line in self.raw_data:
            words = line.strip().split(',')
            words[1:] = [float(x) for x in words[1:]]
            words[0] = int(words[0])
            data.append(words[1:])
            y.append(words[0])

        return np.array(data), np.array(y)


    def trainTestSplit(self):

        #call preProcess
        data, label = self._preProcess()

        #randomly shuffle the data
        new_index = np.random.shuffle(np.arange(len(label)))
        data = data[new_index]
        label = label[new_index]

        train_data = data[:int(len(data)*self.train_test_split)]
        train_label = label[:int(len(data)*self.train_test_split)]
        test_data = data[int(len(data)*self.train_test_split):]
        test_label = label[int(len(data)*self.train_test_split):]

        return train_data,test_data,train_label,test_label

class batchGenerator(object):

    def __init__(self,data,label,batch_size):

        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.batches = None
        self.cursor = 0
        self.num_batches = len(self.label) / self.batch_size

    def createBatches(self,data,label):

        for i in range(self.num_batches):
            self.batches[i] = (data[i*self.batch_size:(i+1)*self.batch_size],label[i*self.batch_size:(i+1)*(self.batch_size)])

    def next(self):
        old_cursor = self.cursor
        self.cursor = (self.cursor+1)/self.num_batches
        return self.batches[old_cursor]

class backblaze(object):

    def __init__(self,dirloc,args):
        '''
        Class to read and load the backblaze 2015 dataset
        '''

        self.dirloc = dirloc
        self.args = args

    def _prune_to_model(self):
        '''
        Read all the csv files and keep only the model data desired
        Returns:

        '''

        if os.path.exists(self.dirloc):

            data = pd.DataFrame([])
            filenamelist = os.listdir(self.dirloc)
            for filename in filenamelist:
                if os.path.isfile(os.path.join(self.dirloc,filename)):
                    t_data = pd.read_csv(os.path.join(self.dirloc,filename))
                    t_data = t_data[t_data['model']==self.args.drive_model]
                    data = data.append(t_data)

            return data

        else:
            raise ValueError("Directory does not exist")


    def _mod_data(self):
        '''
        Based on the provided arguments select the desired columns and pivot appropriate history for each day
        Returns:

        '''

        data = self._prune_to_model()

        data = data[[self.args.keep_cols]]
        data = data.sort_values(['serial_number','date'])






    def _add_labels(self):
        '''
        Based on the provided arguments label the data
        Returns:

        '''
        raise NotImplementedError

    def createBatches(self):
        '''
        Create batches on labeled data by calling all the previous functions
        Returns:

        '''

    def next(self):
        '''
        Return batches
        Returns:

        '''
