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
from sklearn import preprocessing



data_path = "/Users/snehas/data/msd_dataset/YearPredictionMSD.txt"

data = np.loadtxt(data_path,delimiter=',')

low = 1922.0
high = 2011.0

for i in range(0, len(data)):
    data[i][0] = (data[i][0] - low)/(high - low)
x_data = data[:,1:]
y_class = data[:,0:1]
x_data = preprocessing.scale(x_data)
data = np.hstack([y_class, x_data])

train = data[:463715]
test = data[463715:]




np.save('train.npy',train)
np.save('test.npy', test)



