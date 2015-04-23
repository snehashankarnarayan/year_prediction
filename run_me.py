#!/usr/bin/python

from year_prediction import *
import multiprocessing as mp

__author__ = 'snehas'

train_location = "../train.npy"
test_location = "../test.npy"

svr = mp.Process(target=regression, args=("svr", train_location, test_location))
knn = mp.Process(target=regression, args=("knn", train_location, test_location))
tree = mp.Process(target=regression, args=("tree", train_location, test_location))
forest = mp.Process(target=regression, args=("forest", train_location, test_location))

svr.start()
knn.start()
tree.start()
forest.start()