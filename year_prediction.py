#!/usr/bin/python
__author__ = 'snehas'

import numpy as np
import sys
import math
from time import time
import csv
from pprint import pprint
from sklearn import neighbors
from sklearn import svm
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.linear_model import *
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import *
from sklearn.cross_validation import *

__author__ = 'snehas'


TRAIN_DATA = "/Users/snehas/data/msd_dataset/train.npy"
TEST_DATA = "/Users/snehas/data/msd_dataset/test.npy"

#Function to spit the output in an appropriately named CSV file
def output(data):
    with open("output.csv", "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID', 'Target'])
        for i in range(0, len(data)):
            writer.writerow(["{0:0.1f}".format(float(i+1)), "{0:0.1f}".format(data[i])])

def compute_rmse(pred, actual):
    sum = 0.0
    for i in range(0, len(pred)):
        sum = sum + (pred[i] - actual[i])*(pred[i] - actual[i])
    sum = sum/len(pred)
    return math.sqrt(sum)

def regression(method, train_location=TRAIN_DATA, test_location=TEST_DATA):
    train = np.load(train_location)

    #Get the required training data
    x_data = train[:, 1:]  #All but the first column
    y_class = train[:, 0]

    print "Loading data: " + method
    #Get the test data
    test = np.load(test_location)
    test_data = test[:, 1:]

    if(method == "svr"):
        regress = SVR()
    elif(method == "tree"):
        regress = DecisionTreeRegressor()
    elif(method == "forest"):
        regress = ensemble.RandomForestRegressor()
    elif (method == "knn"):
        regress = neighbors.KNeighborsRegressor()


    print "Training"
    regress.fit(x_data, y_class)

    print "Prediction"
    out = regress.predict(test_data)
    #output(out)

    x = compute_rmse(out, test_data[:,0])
    print "RMSE: " + str(x)

    fp = open("output_" + method + ".txt","w")
    fp.write("RMSE: " + method + " :" + str(x))
    fp.close()




if __name__ == "__main__":
    t1 = time()
    regression(sys.argv[1])
    t2 = time()
    print 'Time taken in seconds: %f' % (t2 - t1)