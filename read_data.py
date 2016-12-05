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


class blackblazeReader(object):

    def __init__(self,args):
        '''
        Class to read and load the backblaze 2015 dataset
        '''

        self.args = args

    def _prune_to_model(self):
        '''
        Read all the csv files and keep only the model data desired
        Returns:

        '''

        if os.path.exists(self.args.dirloc):

            data = pd.DataFrame([])
            filenamelist = os.listdir(self.args.dirloc)
            for filename in filenamelist:
                print filename, filename[-3:]
                if os.path.isfile(os.path.join(self.args.dirloc,filename)) and (filename[-3:]=='csv'):
                    t_data = pd.read_csv(os.path.join(self.args.dirloc,filename))
                    t_data = t_data[t_data['model']==self.args.drive_model]
                    data = data.append(t_data)
            return data
        else:
            raise ValueError("Directory does not exist")


    def _mod_data(self,data):
        '''
        Based on the provided arguments select the desired columns and pivot appropriate history for each day
        Returns:

        '''

        data = data.sort_values(['serial_number','date'])
        serialList = data['serial_number'].drop_duplicates().values.tolist()

        res_data = []
        res_label = []
        for serial in serialList:
            t_data = data[data['serial_number']==serial]
            res_label += t_data['failure'].values.tolist()
            t_data = t_data.drop(['serial_number','date','model','capacity','failure'],axis=1)
            row,col = t_data.shape
            for i in range(self.args.hist,row):
                res_data.append(t_data.ix[i-self.args.hist:i].values.flatten())

        res_data = np.array(res_data)
        res_label = np.array(res_label)

        # assume failure post last failure date does not happen to simplify calculation
        for i in range(len(res_label)):
            r_end = min(len(res_label),i+self.args.pred_window)
            res_label[i] = sum(res_label[i:r_end])

        return np.hstack((res_data,res_label))


    def train_test_split(self,split,r_seed=None):
        '''
        Randomly split the
        Args:
            split: list containing ratio's of train, val and test splits
            r_seed: random seed to be initialized

        Returns:
        returns training, validation and testing sets
        '''

        data = self._prune_to_model()

        data_serial_label = data[['serial_number','failure']].drop_duplicates()
        data_serial_label = data_serial_label.groupby('serial_number')['failure'].sum().reset_index()

        if r_seed == None:
            r_seed = 42

        np.random.seed(r_seed)
        idx_perm = np.random.permutation(np.linspace(0,len(data_serial_label)-1,len(data_serial_label)))
        data_serial_label_perm = data_serial_label.ix[idx_perm]

        train_serial_num = data_serial_label_perm.ix[0:int(split[0]*len(data_serial_label_perm))]
        val_serial_num = data_serial_label_perm.ix[int(split[0]*len(data_serial_label_perm)):int(split[0]*len(data_serial_label_perm))+int(split[1]*len(data_serial_label_perm))]
        test_serial_num = data_serial_label_perm.ix[int(split[0]*len(data_serial_label_perm))+int(split[1]*len(data_serial_label_perm)):]

        # count statistics of failures
        print "Training data statistics on failures and non-failures: "
        print train_serial_num.groupby('failure')['serial_number'].size()
        print "Validation data statistics on failures and non-failures: "
        print val_serial_num.groupby('failure')['serial_number'].size()
        print "Testing data statistics on failures and non-failures: "
        print test_serial_num.groupby('failure')['serial_number'].size()

        train = self._mod_data(data[data['serial_number'].isin(train_serial_num['serial_number'].values.tolist())])
        val = self._mod_data(data[data['serial_number'].isin(val_serial_num['serial_number'].values.tolist())])
        test = self._mod_data(data[data['serial_number'].isin(test_serial_num['serial_number'].values.tolist())])

        return train,val,test


class batchGenerator(object):

    def __init__(self,data,label,batch_size):

        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.batches = None
        self.cursor = 0
        self.num_batches = len(self.label) / self.batch_size

    def createBatches(self):

        for i in range(self.num_batches):
            self.batches[i] = (self.data[i*self.batch_size:(i+1)*self.batch_size],self.label[i*self.batch_size:(i+1)*(self.batch_size)])

        return self.num_batches

    def next(self):
        old_cursor = self.cursor
        self.cursor = (self.cursor+1)/self.num_batches
        return self.batches[old_cursor]
