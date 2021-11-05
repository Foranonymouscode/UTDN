"""
   Contains all the utility functions that would be needed
   1. _normalized
   2. _split
   3._batchify
   4. get_batches
    """


import torch
import numpy as np
from torch.autograd import Variable

import read_data

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, data, station_idx,model_type,train, valid, horizon, window, device,normalize=2,prelen=24):
        self.P = window
        self.h = horizon
        self.station_indx = station_idx
        self.model_type = model_type
        self.rawdat = np.concatenate(data,axis=0)
        self.dat = np.zeros(self.rawdat.shape)
        self.original_rows, self.m = self.dat.shape
        # self.predicted = 7
        self.predicted = self.m
        self.prelen = prelen
        self.normalize = 2
        self.scale = np.ones(self.m)
        # self.scale = np.ones(self.predicted)
        self._normalized(normalize)
        self._split(train,  valid)

        self.scale = torch.from_numpy(self.scale).float()
        # tmp = self.test[1].view(self.test[1].size(0)*self.test[1].size(1),self.test[1].size(2)) * self.scale.expand(self.test[1].size(0)*self.test[1].size(1), self.test[1].size(2))
        # tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        # self.scale = self.scale
        self.scale = Variable(self.scale)

        # self.rse = normal_std(tmp)
        # self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))
        # if (normalize == 2):
        #     for i in range(self.m):
        #         if i <self.predicted:
        #             self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
        #         self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train_rate, valid_rate):
        one_station = self.dat.reshape((self.original_rows//35,35,self.m),order='F')
        # print(one_station[0,:,0]*1004,self.dat[0,0]*1004,self.dat[8344,0]*1004)
        orig_rows_one, _, _ = one_station.shape
        train = int(train_rate*orig_rows_one)
        valid = int((train_rate + valid_rate) * orig_rows_one)
        test = orig_rows_one
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, test-self.prelen)

        self.train = self._batchify(train_set, self.h,one_station)
        self.valid = self._batchify(valid_set, self.h,one_station)
        self.test = self._batchify(test_set, self.h,one_station)




    def _batchify(self, idx_set, horizon,one_station):
        #8344,35,F
        n = len(idx_set)
        X = torch.zeros((n, self.P, 35, self.m))
        # Y = torch.zeros((n, 35, self.predicted))
        Y = torch.zeros((n, self.prelen, 35, self.predicted))
        for i in range(n):
            end = idx_set[i]
            start = end - self.P
            X[i, :, :,:] = torch.from_numpy(one_station[start:end, :,:])
            Y[i,:, :,:] = torch.from_numpy(one_station[end:end+self.prelen, :,:])
        return [X, Y]

    def _batchifyCNN(self, idx_set,one_station):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.predicted))
        for i in range(n):
            end = idx_set[i]
            start = end - self.P
            X[i, :, :] = torch.from_numpy(one_station[start:end, :])
            Y[i, :self.predicted] = torch.from_numpy(one_station[idx_set[i]-1, :self.predicted])
            # test1 = X[i,-1,:]
            # test2 = Y[i,:]
            X[i, -1, :] = 0
        # #X = n,72,11(72,11=0)-->n,3,24,11  Y = n,11
        # X = X.unsqueeze(1)
        # X = torch.reshape(X,(X.size(0),3,24,self.m))
        # print(X[0,-1,-1,:])
        #X = n,72,11 --> n,11,72 -->n,11,3,24
        X = X.transpose(2,1)
        X = X.unsqueeze(2)
        X = torch.reshape(X,(X.size(0),self.m,-1,24))
        # print(X[0,:,-1,-1])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    def get_batches_filter(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)

            all_day = X[:, 0, :, :-1] * self.scale[0]
            target_day = X[:, 0, -1, :-1].unsqueeze(1)  # B,D,23-->B,1,23
            target_day = target_day.expand(-1, all_day.size(1), -1) * self.scale[0]
            dayatt_score = torch.cosine_similarity(all_day, target_day,dim=2)
            score,indice = dayatt_score.sort(dim=1)

            ones = torch.ones_like(dayatt_score)
            zeros = torch.zeros_like(dayatt_score)
            dayfilter = torch.where(dayatt_score < 0.98, zeros, ones) #B,7
            X = torch.mul(X[:,0], dayfilter.unsqueeze(2))

            index_x = torch.arange(0, X.size(0),1).unsqueeze(1).expand(X.size(0),7)
            sorted_X = X[index_x,indice]
            X = sorted_X[:,-3:,:].unsqueeze(1) #B,2,24-->B,1,2,24
            Y = Y[:,0]
            print(len(torch.where(score[:,-2] > 0.98)[0])/512)

            yield Variable(X), Variable(Y)
            start_idx += batch_size

# Data = Data_utility(read_data.read_df(), 0.6, 0.2, 2, 24, 2)
# print(Data.train)