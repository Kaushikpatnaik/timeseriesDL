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
            t_label = t_data['failure'].values.tolist()
            t_data = t_data.drop(['serial_number','date','model','capacity_bytes','failure'],axis=1).values
            row,col = t_data.shape
            for i in range(self.args.hist,row):
                #print t_data[i-self.args.hist:i,:].flatten()
                res_label.append(t_label[i])
                res_data.append(t_data[i-self.args.hist:i+1,:].flatten())

        res_data = np.array(res_data)
        res_label = np.array(res_label).reshape(len(res_label),1)

        # assume failure post last failure date does not happen to simplify calculation
        res_label_final = np.zeros((len(res_label),2))
        for i in range(len(res_label)):
            r_end = min(len(res_label),i+self.args.pred_window)
            t_label = sum(res_label[i:r_end])
            if t_label == 0:
                res_label_final[i] = (1,0)
            else:
                res_label_final[i] = (0,1)
        res_label_final = np.array(res_label_final)

        print "Feature and Label Shape: "
        print res_data.shape, res_label_final.shape

        return np.concatenate((res_data,res_label_final),axis=1)


    def train_test_split(self,split,r_seed=None):
        '''
        Randomly split the
        Args:
            split: list containing ratio's of train, val and test splits
            r_seed: random seed to be initialized

        Returns:
        returns training, validation and testing sets
        '''

        print "Split provided: "
        print split

        data = self._prune_to_model()

        data_serials = data['serial_number'].drop_duplicates()

        print "Overall Failure Statistics: "
        print data.groupby('failure')['serial_number'].size().reset_index()

        data_serial_label = data.groupby('serial_number')['failure'].sum().reset_index()

        assert(len(data_serials)==len(data_serial_label))

        if r_seed == None:
            r_seed = 42

        np.random.seed(r_seed)
        idx_perm = np.random.permutation(np.linspace(0,len(data_serial_label)-1,len(data_serial_label)))
        data_serial_label_perm = data_serial_label.ix[idx_perm].reset_index()

        assert(len(data_serial_label)==len(data_serial_label_perm))

        #print "Permuted Serial Indexes: "
        #print data_serial_label_perm

        print "Checking ranges for split: "
        print 0, int(split[0]*len(data_serial_label_perm))
        print int(split[0]*len(data_serial_label_perm)), np.floor((split[0]+split[1])*len(data_serial_label_perm))
        print np.ceil(sum(split[:2])*len(data_serial_label_perm)), len(data_serial_label_perm)

        train_serial_num = data_serial_label_perm.ix[0:int(np.floor(split[0]*len(data_serial_label_perm)))]
        val_serial_num = data_serial_label_perm.ix[int(split[0]*len(data_serial_label_perm)):int(np.floor((split[0]+split[1])*len(data_serial_label_perm)))]
        test_serial_num = data_serial_label_perm.ix[int(np.ceil(sum(split[:2])*len(data_serial_label_perm))):]

        # count statistics of failures
        print "Training data statistics on failures and non-failures: "
        print train_serial_num.groupby('failure')['serial_number'].size()
        print "Validation data statistics on failures and non-failures: "
        print val_serial_num.groupby('failure')['serial_number'].size()
        print "Testing data statistics on failures and non-failures: "
        print test_serial_num.groupby('failure')['serial_number'].size()

        # check that no serial number exists in both groups
        print "Do the train and validation sets overlap ?"
        print sum([int(x==y) for x in train_serial_num['serial_number'].values.tolist() for y in val_serial_num['serial_number'].values.tolist()]) > 0

        print "Do the train and test sets overlap ?"
        print sum([int(x==y) for x in train_serial_num['serial_number'].values.tolist() for y in test_serial_num['serial_number'].values.tolist()]) > 0

        print "Do the test and validation sets overlap ?"
        print sum([int(x==y) for x in test_serial_num['serial_number'].values.tolist() for y in val_serial_num['serial_number'].values.tolist()]) > 0

        train = self._mod_data(data[data['serial_number'].isin(train_serial_num['serial_number'].values.tolist())])
        val = self._mod_data(data[data['serial_number'].isin(val_serial_num['serial_number'].values.tolist())])
        test = self._mod_data(data[data['serial_number'].isin(test_serial_num['serial_number'].values.tolist())])

        return train,val,test


class batchGenerator(object):

    def __init__(self,data,batch_size,op_channels):

        self.data = data
        self.batch_size = batch_size
        self.op_channels = op_channels
        self.cursor = 0
        self.num_batches = len(self.data) / self.batch_size
        self.batches = np.zeros((self.num_batches,self.batch_size, self.data.shape[1]), dtype=np.int32)

    def createBatches(self):

        for i in range(self.num_batches):
            self.batches[i] = self.data[i*self.batch_size:(i+1)*self.batch_size,:]

    def get_num_batches(self):
        return self.num_batches

    def get_ip_channels(self):
        return self.data.shape[1]-self.op_channels

    def next(self):
        old_cursor = self.cursor
        self.cursor = (self.cursor+1)/self.num_batches
        return (self.batches[old_cursor,:,0:-self.op_channels], self.batches[old_cursor,:,-self.op_channels:])
