#!/usr/bin/python
__author__ = 'snehas'

import numpy as np
import sys
import math
from time import time
import csv
from pprint import pprint
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.linear_model import *
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import *
from sklearn.cross_validation import *



TRAIN_DATA = "/Users/snehas/data/msd_dataset/train_sample.npy"
TEST_DATA = "/Users/snehas/data/msd_dataset/test_sample.npy"

train = np.load(TRAIN_DATA)
x_data = train[:, 1:]  #All but the first column
y_class = train[:, 0]
datasets.dump_svmlight_file(x_data,y_class, "train_sample_svmlight", zero_based=False)

test = np.load(TEST_DATA)
x_data = test[:, 1:]  #All but the first column
y_class = test[:, 0]
datasets.dump_svmlight_file(x_data,y_class, "test_sample_svmlight", zero_based=False)





