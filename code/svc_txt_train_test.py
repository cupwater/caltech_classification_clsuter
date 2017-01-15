import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import numpy as np


train_datafile = 'train_data.txt'
test_datafile = 'test_data.txt'
train_labelfile = 'train_label.txt'
test_labelfile = 'test_label.txt'

def load_data(datafile, labelfile):
    # train_labels = np.loadtxt(train_labelfile)
    # train_labels = train_labels.astype(int)
    test_data = np.loadtxt(datafile)
    test_labels = np.loadtxt(labelfile)
    test_labels = test_labels.astype(int)
    # return train_data, train_labels, test_data, test_labels
    return test_data, test_labels


#read data from lmdb file
x_test, y_test = load_data(test_datafile, test_labelfile)
x_train, y_train = load_data(train_datafile, train_labelfile)

h = .02  # step size in the mesh
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)
#lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)
# joblib.dump(svc, "svc_model.m")
# joblib.dump(rbf_svc, "rbf_svc_model.m")
# joblib.dump(poly_svc, "poly_svc_model.m")
# joblib.dump(lin_svc, "lin_svc_model.m")

# svc      = joblib.load("svc_model.m")
# rbf_svc  = joblib.load("rbf_svc_model.m")
# poly_svc = joblib.load("poly_svc_model.m")
# lin_svc  = joblib.load("lin_svc_model.m")
test_num = 2690
#for k, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
if test_num == 2690:
    correct_num = 0
    false_num = 0
    for i in range(test_num):
        predict_y = svc.predict(x_test[i])
        if predict_y != y_test[i]:
            false_num += 1
        else :
            correct_num += 1

    print false_num, correct_num

