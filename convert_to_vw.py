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



TRAIN_DATA = "/Users/snehas/data/msd_dataset/train.npy"
TEST_DATA = "/Users/snehas/data/msd_dataset/test.npy"


train_file = open("train_dataset", "w")
test_file = open("test_dataset", "w")

train = np.load(TRAIN_DATA)
test = np.load(TEST_DATA)



lines = list()

for elem in train:
    line = str(elem[0]) + " | "
    for i in range(1,len(elem)):
        line = line + " " + str(elem[i])
    line = line + '\n'
    lines.append(line)

train_file.writelines(lines)

lines = list()
for elem in test:
    line = str(elem[0]) + " | "
    for i in range(1,len(elem)):
        line = line + " " + str(elem[i])
    line = line + '\n'
    lines.append(line)

test_file.writelines(lines)









