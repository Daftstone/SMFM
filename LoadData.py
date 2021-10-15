import numpy as np
import os


class LoadData(object):
    '''given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type):
        in_data = ['industrial', 'sport', 'elect', 'beauty', 'book', 'pet', 'automotive', 'yelp']
        if (dataset in in_data):
            data = np.load('data/%s/x.npy' % dataset)
            label = np.load('data/%s/y.npy' % dataset)
            if (loss_type != 'log_loss'):
                label = label * 2 - 1.
            self.features_M = np.max(data) + 1
            train_index = np.arange(int(len(data) * 0.7))
            val_index = np.arange(int(len(data) * 0.7), int(len(data) * 0.9))
            test_index = np.arange(int(len(data) * 0.9), int(len(data)))
            self.Train_data = {'X': data[train_index], 'Y': label[train_index]}
            self.Validation_data = {'X': data[val_index], 'Y': label[val_index]}
            self.Test_data = {'X': data[test_index], 'Y': label[test_index]}
            self.Train_data['weights'] = np.ones_like(np.array(self.Train_data['X']))
            self.Validation_data['weights'] = np.ones_like(np.array(self.Validation_data['X']))
            self.Test_data['weights'] = np.ones_like(np.array(self.Test_data['X']))
            print("# of data:", len(data))
        elif (dataset == 'criteo' or dataset == 'taobao' or dataset == 'apply' or dataset == 'enzymes'):
            data = np.load('data/%s/x.npy' % dataset)
            label = np.load('data/%s/y.npy' % dataset)
            weights = np.load('data/%s/weights.npy' % dataset)
            print(np.mean(label))
            if (loss_type != 'log_loss'):
                label = label * 2 - 1.
            self.features_M = np.max(data) + 1
            train_index = np.arange(int(len(data) * 0.7))
            val_index = np.arange(int(len(data) * 0.7), int(len(data) * 0.9))
            test_index = np.arange(int(len(data) * 0.9), int(len(data)))
            self.Train_data = {'X': data[train_index], 'Y': label[train_index]}
            self.Validation_data = {'X': data[val_index], 'Y': label[val_index]}
            self.Test_data = {'X': data[test_index], 'Y': label[test_index]}
            self.Train_data['weights'] = weights[train_index]
            self.Validation_data['weights'] = weights[val_index]
            self.Test_data['weights'] = weights[test_index]
            print("# of data:", len(data))
        else:
            self.path = path + dataset + "/"
            self.trainfile = self.path + dataset + ".train.libfm"
            self.testfile = self.path + dataset + ".test.libfm"
            self.validationfile = self.path + dataset + ".validation.libfm"
            self.features_M = self.map_features()
            self.Train_data, self.Validation_data, self.Test_data = self.construct_data(loss_type)
            self.Train_data['weights'] = np.ones_like(np.array(self.Train_data['X']))
            self.Validation_data['weights'] = np.ones_like(np.array(self.Validation_data['X']))
            self.Test_data['weights'] = np.ones_like(np.array(self.Test_data['X']))

    def map_features(self):  # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        # print("features_M:", len(self.features))
        return len(self.features)

    def read_features(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.features:
                    self.features[item] = i
                    i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        X_, Y_, Y_for_logloss = self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        print("# of training:", len(Y_))

        X_, Y_, Y_for_logloss = self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        print("# of validation:", len(Y_))

        X_, Y_, Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        print("# of test:", len(Y_))

        return Train_data, Validation_data, Test_data

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open(file)
        X_ = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append(1.0 * float(items[0]))

            if float(items[0]) > 0:  # > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append(v)

            X_.append([self.features[item] for item in items[1:]])
            line = f.readline()
        f.close()
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [Y_[i] for i in indexs]
        Data_Dic['X'] = [X_[i] for i in indexs]
        return Data_Dic

    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in range(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in range(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in range(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in range(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable
