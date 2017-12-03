import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.examples.tutorials.mnist import input_data
import pickle
from sklearn.preprocessing import MinMaxScaler
import logging

module_logger = logging.getLogger('timeSeriesDL.read_data')

class ucrDataReader(object):
    '''
    UCR time series has only one channel
    '''

    def __init__(self,raw_data,train_test_split,op_classes):

        self.raw_data = raw_data
        self.train_test_split = train_test_split
        self.op_classes = op_classes


    def _preProcess(self):

        data = []
        y = []
        for line in self.raw_data:
            words = line.strip().split(',')
            words[1:] = [float(x) for x in words[1:]]
            words[0] = int(words[0])
            data.append(words[1:])
            y.append(words[0])

        return np.array(data), np.array(y).reshape(len(y),1)


    def trainTestSplit(self):

        #call preProcess
        data, label = self._preProcess()

        #randomly shuffle the data
        np.random.seed(42)
        new_index = np.random.permutation(len(label))
        data_shuff = data[new_index,:]
        label_shuff = label[new_index,:]

        label_onehot = np.zeros((len(label),self.op_classes))
        for i in range(len(label)):
            label_onehot[i][label[i]-1] = 1

        train_data = data[:int(len(data)*self.train_test_split[0]),:]
        train_label = label_onehot[:int(len(data)*self.train_test_split[0]),:]

        val_data = data[int(len(data)*self.train_test_split[0]):int(len(data)*self.train_test_split[1]),:]
        val_label = label_onehot[int(len(data)*self.train_test_split[0]):int(len(data)*self.train_test_split[1]),:]

        test_data = data[int(len(data)*self.train_test_split[1]):,:]
        test_label = label_onehot[int(len(data)*self.train_test_split[1]):,:]

        module_logger.info("Train, Val, Test data shape: ")
        module_logger.info(train_data.shape, val_data.shape, test_data.shape)

        module_logger.info("Train, Val, Test label shape: ")
        module_logger.info(train_label.shape, val_label.shape, test_label.shape)

        train = np.concatenate((train_data,train_label),axis=1)
        val = np.concatenate((val_data,val_label),axis=1)
        test = np.concatenate((test_data,test_label),axis=1)

        return train, val, test

class phmReader(object):

    def __init__(self,args):
        '''
        Class to read and load the phm08 dataset
        '''

        self.args = args

    def _mode_to_data(self):
        raise NotImplementedError

class blackblazeReader(object):

    def __init__(self,dirloc,drive_model,hist,pred_window,year):
        '''
        Class to read and load the backblaze raw_data dataset
        '''

        self.dirloc = dirloc
        self.hist = hist
        self.pred_window = pred_window
        self.drive_model = drive_model
        self.data_year = year

    def _prune_to_model(self):
        '''
        Read all the csv files and keep only the model data desired
        Returns:

        '''

        if os.path.exists(self.dirloc):

            data = pd.DataFrame([])
            stats = pd.DataFrame([],columns=['model','serial_number','failure'])
            filenamelist = os.listdir(self.dirloc)
            for idx, filename in enumerate(filenamelist):
                if idx%50 == 0:
                    print(filename, filename[-3:])
                if os.path.isfile(os.path.join(self.dirloc,filename)) and (filename[-3:]=='csv'):
                    if filename.split('-')[0] == self.data_year:
                        t_data = pd.read_csv(os.path.join(self.dirloc,filename))
                        t2_data = t_data[['model','serial_number','failure']].drop_duplicates()
                        t_data = t_data[t_data['model']==self.drive_model]
                        data = data.append(t_data)
                        stats = stats.append(t2_data)

            module_logger.info(stats.groupby(['model','failure'])['serial_number'].size().reset_index())

            return data
        else:
            raise ValueError("Directory does not exist")


    def _mod_data(self,data):
        '''
        Based on the provided arguments select the desired columns and pivot appropriate history for each day
        List of parameters provided in the paper:

        Returns:

        '''

        data = data.sort_values(['serial_number','date'])
        serialList = data['serial_number'].drop_duplicates().values.tolist()

        res_data = []
        res_label = []
        for serial in serialList:
            t_data = data[data['serial_number']==serial].sort_values('date')
            t_label = t_data['failure'].values.tolist()
            t_data = t_data[['smart_1_raw','smart_5_raw','smart_7_raw','smart_183_raw','smart_184_raw','smart_187_raw',\
                             'smart_188_raw','smart_189_raw','smart_190_raw','smart_193_raw','smart_194_raw','smart_197_raw',\
                             'smart_198_raw','smart_199_raw','smart_240_raw','smart_241_raw','smart_242_raw']].interpolate(method='linear').values
            '''
            Difference features, keep away for now
            t_data_diff = t_data.diff(1).fillna(0)
            t_data_diff.columns = ['smart_5_raw_diff','smart_183_raw_diff','smart_184_raw_diff','smart_187_raw_diff','smart_188_raw_diff','smart_193_raw_diff','smart_197_raw_diff']
            t_data = pd.concat([t_data,t_data_diff],axis=1).values
            '''
            row,col = t_data.shape
            for i in range(self.hist,row):
                #print t_data[i-self.args.hist:i,:].flatten()
                res_label.append(t_label[i])
                res_data.append(t_data[i-self.hist:i,:].flatten())

        res_data = np.array(res_data)
        res_label = np.array(res_label).reshape(len(res_label),1)

        # assume failure post last failure date does not happen to simplify calculation
        res_label_final = np.zeros((len(res_label),2))
        for i in range(len(res_label)):
            r_end = min(len(res_label),i+self.pred_window)
            t_label = sum(res_label[i:r_end])
            if t_label == 0:
                res_label_final[i] = (1,0)
            else:
                res_label_final[i] = (0,1)
        res_label_final = np.array(res_label_final)

        module_logger.info("Feature and Label Shape: ")
        module_logger.info((res_data.shape, res_label_final.shape))

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

        module_logger.info("Split provided: ")
        module_logger.info(split)

        data = self._prune_to_model()

        data_serials = data['serial_number'].drop_duplicates()

        data_serial_label = data.groupby('serial_number')['failure'].sum().reset_index()

        module_logger.info("Overall Failure Statistics: ")
        module_logger.info(data_serial_label.groupby('failure')['serial_number'].size().reset_index())

        assert(len(data_serials)==len(data_serial_label))

        if r_seed == None:
            r_seed = 42

        np.random.seed(r_seed)
        idx_perm = np.random.permutation(np.linspace(0,len(data_serial_label)-1,len(data_serial_label)))
        data_serial_label_perm = data_serial_label.ix[idx_perm].reset_index()

        assert(len(data_serial_label)==len(data_serial_label_perm))

        #print "Permuted Serial Indexes: "
        #print data_serial_label_perm

        module_logger.info("Checking ranges for split: ")
        module_logger.info(int(split[0]*len(data_serial_label_perm)))
        module_logger.info((int(split[0]*len(data_serial_label_perm)), np.floor((split[0]+split[1])*len(data_serial_label_perm))))
        module_logger.info((np.ceil(sum(split[:2])*len(data_serial_label_perm)), len(data_serial_label_perm)))

        train_serial_num = data_serial_label_perm.ix[0:int(np.floor(split[0]*len(data_serial_label_perm)))]
        val_serial_num = data_serial_label_perm.ix[int(split[0]*len(data_serial_label_perm)):int(np.floor((split[0]+split[1])*len(data_serial_label_perm)))]
        test_serial_num = data_serial_label_perm.ix[int(np.ceil(sum(split[:2])*len(data_serial_label_perm))):]

        # count statistics of failures
        module_logger.info("Training data statistics on failures and non-failures: ")
        module_logger.info(train_serial_num.groupby('failure')['serial_number'].size())
        module_logger.info("Validation data statistics on failures and non-failures: ")
        module_logger.info(val_serial_num.groupby('failure')['serial_number'].size())
        module_logger.info("Testing data statistics on failures and non-failures: ")
        module_logger.info(test_serial_num.groupby('failure')['serial_number'].size())

        # check that no serial number exists in both groups
        module_logger.info("Do the train and validation sets overlap ?")
        module_logger.info(sum([int(x==y) for x in train_serial_num['serial_number'].values.tolist() for y in val_serial_num['serial_number'].values.tolist()]) > 0)

        module_logger.info("Do the train and test sets overlap ?")
        module_logger.info(sum([int(x==y) for x in train_serial_num['serial_number'].values.tolist() for y in test_serial_num['serial_number'].values.tolist()]) > 0)

        module_logger.info("Do the test and validation sets overlap ?")
        module_logger.info(sum([int(x==y) for x in test_serial_num['serial_number'].values.tolist() for y in val_serial_num['serial_number'].values.tolist()]) > 0)

        train = self._mod_data(data[data['serial_number'].isin(train_serial_num['serial_number'].values.tolist())])
        val = self._mod_data(data[data['serial_number'].isin(val_serial_num['serial_number'].values.tolist())])
        test = self._mod_data(data[data['serial_number'].isin(test_serial_num['serial_number'].values.tolist())])

        return train,val,test

def get_data_obj(args,data_opt):
    '''
    Reading the data and flatting the sequence. i.e each row is seq_len (history) * num of channels (features)
    Args:
        args:

    Returns:

    '''

    # TODO: determine the dataset num classes automatically
    if data_opt == 'backblaze':

        dirloc = './data/backblaze/raw_data/'

        try:
            train_data = pickle.load(open('./data/backblaze/processed_data/' + str(args['drive_model']) + '_' + str(args['year']) + '_' +
                                         str(args['hist']) + '_' + str(args['pred_window']) + '_train.pkl','rb'))
            val_data = pickle.load(open('./data/backblaze/processed_data/' + str(args['drive_model']) + '_' + str(args['year']) + '_' +
                                         str(args['hist']) + '_' + str(args['pred_window']) + '_val.pkl','rb'))
            test_data = pickle.load(open('./data/backblaze/processed_data/' + str(args['drive_model']) + '_' + str(args['year']) + '_' +
                                         str(args['hist']) + '_' + str(args['pred_window']) + '_test.pkl','rb'))
        except:
            backblaze_data = blackblazeReader(dirloc, args['drive_model'], args['hist'], args['pred_window'], args['year'])
            train_data, val_data, test_data = backblaze_data.train_test_split(args['split_ratio'])
            print("Saving the datasets")

            # Saved the train, val and test sets for future work, as they take a lot of time to prepare
            pickle.dump(train_data, open('./data/backblaze/processed_data/' + str(args['drive_model']) + '_' + str(args['year']) + '_' +
                                         str(args['hist']) + '_' + str(args['pred_window']) + '_train.pkl','wb'))
            pickle.dump(val_data, open('./data/backblaze/processed_data/' + str(args['drive_model']) + '_' + str(args['year']) + '_' +
                                         str(args['hist']) + '_' + str(args['pred_window']) + '_val.pkl','wb'))
            pickle.dump(test_data, open('./data/backblaze/processed_data/' + str(args['drive_model']) + '_' + str(args['year']) + '_' +
                                         str(args['hist']) + '_' + str(args['pred_window']) + '_test.pkl','wb'))

        op_channels = 2
        seq_len = args['hist']
        ip_channels = int((train_data.shape[1] - op_channels)/seq_len)

        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

    elif data_opt == 'phm08':
        raise NotImplementedError

    else:
        raise ValueError("Dataset option provided does not exist")

    return train_data, val_data, test_data, ip_channels, op_channels, seq_len

class batchGenerator(object):

    def __init__(self,data,batch_size,ip_channels,op_channels,seq_len):

        self.data = data
        self.batch_size = batch_size
        self.ip_channels = ip_channels
        self.op_channels = op_channels
        self.seq_len = seq_len
        self.cursor = 0
        self.num_batches = int(len(self.data) / self.batch_size)

    def get_num_batches(self):
        return self.num_batches

    def get_data_size(self):
        return self.data.shape

    def next(self):
        temp = self.data[self.cursor*self.batch_size:(self.cursor+1)*self.batch_size,:]
        x = temp[:,0:-self.op_channels]
        y = temp[:,-self.op_channels:]
        x = np.reshape(x,[self.batch_size,self.seq_len,self.ip_channels])

        if self.cursor + 1 > self.num_batches:
            self.data = np.random.permutation(self.data)
        self.cursor = (self.cursor+1)%self.num_batches

        return x,y

class balBatchGenerator(object):

    def __init__(self,data,batch_size,ip_channels,op_channels,seq_len,label_ratio):

        self.batch_size = batch_size
        self.ip_channels = ip_channels
        self.op_channels = op_channels
        self.seq_len = seq_len
        self.cursor = 0
        self.label_ratio = label_ratio

        # determine unbalanced ratio
        labels = np.argmax(data[:,-self.op_channels:],axis=1)
        self.uniq_labels, uniq_idx, label_count = np.unique(labels,return_inverse=True,return_counts=True)
        module_logger.info("Labels: "+ ",".join([str(x) for x in self.uniq_labels]))
        module_logger.info("Counts: "+ ",".join([str(x) for x in label_count]))

        # for given batch_size determine size of each label
        self.label_ratio_batch = {}
        for label,ratio in label_ratio.items():
            self.label_ratio_batch[label] = int(np.floor(ratio*self.batch_size))
        diff = self.batch_size - sum(self.label_ratio_batch.values())
        module_logger.info("Ratios: " + ",".join([str(x) for x in self.label_ratio_batch.keys()]))
        module_logger.info("Ratios: " + ",".join([str(x) for x in self.label_ratio_batch.values()]))

        i = 0
        while diff > 0:
            label = list(self.label_ratio_batch.keys())[i]
            self.label_ratio_batch[label] += 1
            diff -= 1
            i += 1
        module_logger.info("Adjusted Ratios: " + ",".join([str(x) for x in self.label_ratio_batch.keys()]))
        module_logger.info("Adjusted Ratios: " + ",".join([str(x) for x in self.label_ratio_batch.values()]))

        self.label_idx_data = {}
        self.label_counter = {}
        for i in self.uniq_labels:
            self.label_idx_data[i] = data[labels==i,:]
            self.label_counter[i] = 0

        self.min_count_class = self.uniq_labels[np.argmax(label_count)]
        self.min_count = min(label_count)

    def next(self):

        ret_batch = np.zeros((self.batch_size,self.ip_channels*self.seq_len+self.op_channels))
        acc_ratio = 0
        for i,ratio in self.label_ratio_batch.items():
            temp = self.label_idx_data[i]
            ret_batch[acc_ratio:acc_ratio+ratio,:] = temp[:ratio,:]
            acc_ratio += ratio
            self.label_counter[i] += ratio

        x = ret_batch[:,0:-self.op_channels]
        y = ret_batch[:,-self.op_channels:]
        x = np.reshape(x,[self.batch_size,self.seq_len,self.ip_channels])

        # TODO: It's possible this mechanism results in some observations from the major label being missed
        for key,val in self.label_idx_data.items():
            if self.label_counter[key] + 1 > len(val):
                self.label_idx_data[key] = np.random.permutation(self.label_idx_data[key])
                self.label_counter[key] = 0

        #labels_batch = np.argmax(y,axis=1)
        #batch_uniq_labels, batch_label_count = np.unique(labels_batch, return_counts=True)
        #print batch_uniq_labels, batch_label_count

        return x,y

def low_pass_and_subsample(data,wind_len=[5,10,15],sample_rate=[2,3,4]):
    '''
    Run a moving average filter, and sub sample the data channel wise to produce 6 transformations of the data
    :param data: original data
    :param wind_len: list specifying window length for moving average
    :param sample_rate: list specifying rate of sub-sampling the data
    :return: data_new, with transformations appended
    '''

    data_new = data.copy(deep=True)
    for i in range(len(wind_len)):
        for j in range(data.shape[1]):
            data_new[:,j] = np.hstack((data_new[:,j],np.convolve(data[:,j],np.ones((wind_len[i]))/wind_len[i],'same')))

    sub_sampled_lens = []
    for i in range(len(sample_rate)):
        for j in range(data.shape[1]):
            temp = data[::,sample_rate[i]].copy()
            sub_sampled_lens.append(len(temp))
            data_new[:,j] = np.hstack((data_new[:,j],temp))

    return data_new, sub_sampled_lens

def freq_transform(data):
    '''
    Return the dct of power spectrum of the time series signal.
    Closely analogous to MFCC computation for speech signals
    :return:
    '''

    data_fft = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_fft[i,:] = np.fft.fft(data[:,i]).real

    return data_fft












