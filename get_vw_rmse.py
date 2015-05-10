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
from sklearn import grid_search
from sklearn.cross_validation import *

def compute_rmse(pred, actual):
    return np.sqrt(((pred - actual) ** 2).mean())

test_location = "/Users/snehas/data/msd_dataset/test_sample.npy"
#pred_location = "/Users/snehas/vagroot/shared_files/data/test.pred"
pred_location = "/Users/snehas/Downloads/svm_light_osx.8.4_i7/svm_out.txt"
test = np.load(test_location)
pred = open(pred_location,"r")
test_values = test[:,0]
pred_values = np.zeros(len(test_values))
j = 0
for line in pred:
    pred_values[j] = float(line)
    j = j + 1

pred_values = pred_values.T

print compute_rmse(pred_values, test_values )

