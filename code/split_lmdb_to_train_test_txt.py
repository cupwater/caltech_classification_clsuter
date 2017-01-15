# -*- coding: utf-8 -*-
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


datafile = 'examples/caltech/caltech_features'
labelfile = 'examples/caltech/labels.txt'
def load_data(datafile, labelfile):
    labels = np.loadtxt(labelfile)
    lmdb_env = lmdb.open(datafile)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    i = 0
    data = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        temp = caffe.io.datum_to_array(datum)
        temp = temp.reshape(4096)
        for i in range(4096):
            temp[i] = round(temp[i], 2)
        data.append(temp)
        i = i+1
    lmdb_env.close()
    return data, labels

#X:含label的数据集：分割成训练集和测试集
#test_size:测试集占整个数据集的比例
def trainTestSplit(data, labels, test_size=0.3):
    last_label = labels[0]#上一个样本的标签
    label_num_array = []  #类别数量数组，记录每个类别的样本数量
    current_label_num = 0 #当前类别的数量
    label_index = 0       #所有样本数组的下标
    label_num_index = 0   #类别数量数组下标
    for current_label in labels:
        if current_label == last_label:
            current_label_num = current_label_num + 1
        else :
            last_label = current_label
            label_num_array.append(current_label_num)
            label_num_index = label_num_index + 1
            current_label_num = 1
    train_data = []
    test_data  = []
    train_label= []
    test_label = []
    train_index = 0 #index of train data array
    test_index  = 0 #index of test  data array
    data_index  = 0 #idnex of data array
    for current_label_num in label_num_array:
        test_num  = int(current_label_num * test_size)
        train_num = int(current_label_num - test_num)
        for i in range(train_num):
            train_data.append(data[data_index])
            train_label.append(int(labels[data_index]))
            train_index = train_index+1
            data_index = data_index+1
        for i in range(test_num):
            test_data.append(data[data_index])
            test_label.append(int(labels[data_index]))
            test_index = test_index+1
            data_index = data_index+1

    np.savetxt("train_data.txt", train_data)
    np.savetxt("train_label.txt", train_label)
    np.savetxt("test_data.txt", test_data)
    np.savetxt("test_label.txt", test_label)
    return train_data, test_data, train_label, test_label

data, labels = load_data(datafile, labelfile)
x_train, x_test, y_train, y_test = trainTestSplit(data, labels, test_size = 0.3)

