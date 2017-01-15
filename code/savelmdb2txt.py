# -*- coding: utf-8 -*-
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

datafile = 'examples/caltech/caltech_features'
labelfile = 'examples/caltech/labels.txt'
def load_data(datafile, labelfile):
    labels = np.loadtxt(labelfile)
    lmdb_env = lmdb.open(datafile)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    data = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        temp = caffe.io.datum_to_array(datum)
        temp = temp.reshape(4096)
        data.append(temp)
    lmdb_env.close()
    return data, labels

#X:含label的数据集：分割成训练集和测试集
#test_size:测试集占整个数据集的比例
def trainTestSplit(data, labels):
    np.savetxt("data.txt", data)
    np.savetxt("label.txt", labels)

data, labels = load_data(datafile, labelfile)
trainTestSplit(data, labels)

