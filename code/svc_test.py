import numpy as np
import time
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

test_datafile  = 'examples/caltech/test_features'
test_labelfile = 'examples/caltech/test_label.txt'

def load_data( test_datafile, test_labelfile):
    test_labels = np.loadtxt(test_labelfile)
    test_labels = test_labels.astype(int)
    lmdb_env = lmdb.open(test_datafile)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    test_data = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        temp = caffe.io.datum_to_array(datum)
        temp = temp.reshape(4096)
        test_data.append(temp)
    lmdb_env.close()
    return test_data, test_labels

#read data from lmdb file
x_test, y_test = load_data(test_datafile, test_labelfile)

svc = joblib.load("svc_model.m")
rbf_svc  = joblib.load("rbf_svc_model.m")
poly_svc = joblib.load("poly_svc_model.m")
lin_svc  = joblib.load("lin_svc_model.m")

svc_predicted      = svc.predict(x_test)
rbf_svc_predicted  = poly_svc.predict(x_test)
poly_svc_predicted = poly_svc.predict(x_test)
lin_svc_predicted  = lin_svc.predict(x_test)

for k, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    start = time.clock()
    predicted = clf.predict(x_test)
    precision_nb=metrics.precision_score(y_test, predicted)
    recall_nb=metrics.recall_score(y_test, predicted)
    accuracy_nb=metrics.accuracy_score(y_test, predicted)
    print precision_nb, recall_nb, accuracy_nb
    end = time.clock()
    print end-start