import numpy as np
import time
from sklearn import svm
from sklearn.externals import joblib
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2


train_datafile = 'examples/caltech/train_features'
train_labelfile = 'examples/caltech/train_label.txt'

def load_data(train_datafile, train_labelfile):
    train_labels = np.loadtxt(train_labelfile)
    train_labels = train_labels.astype(int)
    print train_labels
    lmdb_env = lmdb.open(train_datafile)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    train_data = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        temp = caffe.io.datum_to_array(datum)
        temp = temp.reshape(4096) 
        train_data.append(temp)
    lmdb_env.close()

    # return train_data, train_labels, test_data, test_labels
    return train_data, train_labels

#read data from lmdb file
x_train, y_train = load_data(train_datafile, train_labelfile)
h = .01  # step size in the mesh
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
start = time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
end = time.clock()
print end-start

start = time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train)
end = time.clock()
print end-start

start = time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)
end = time.clock()
print end-start

start = time.clock()
lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)
end = time.clock()
print end-start

joblib.dump(svc, "svc_model.m")
joblib.dump(rbf_svc, "rbf_svc_model.m")
joblib.dump(poly_svc, "poly_svc_model.m")
joblib.dump(lin_svc, "lin_svc_model.m")

