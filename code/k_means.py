#coding=utf-8

from numpy import *
import numpy as np
import time
from sklearn import svm
from sklearn.externals import joblib
import caffe
import lmdb
from caffe.proto import caffe_pb2

data_file = "examples/caltech/caltech_features"
label_file = "examples/caltech/labels.txt"

# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))

# init centroids with random samples
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

# k-means cluster
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = mat(zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)
	iterNum = 0
	while clusterChanged and iterNum < 1000:
		clusterChanged = False
		clusterChangedNumber = 0
		## for each sample
		for i in xrange(numSamples):
			minDist  = 10000000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2
				clusterChangedNumber += 1

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = mean(pointsInCluster, axis = 0)
		iterNum += 1
		print clusterChangedNumber

	print 'Congratulations, cluster complete!'
	return centroids, clusterAssment


def load_data( datafile, labelfile):
    labels = np.loadtxt(labelfile)
    labels = labels.astype(int)
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

    # return train_data, train_labels, test_data, test_labels
    return data, labels

dataset, labels = load_data(data_file, label_file)
dataset = mat(dataset)
start = time.clock()
centroids, clusterAssment = kmeans(dataset, 102)
end = time.clock()
print end-start
np.savetxt("center.txt", centroids)
np.savetxt("clusterAssment.txt", clusterAssment)
