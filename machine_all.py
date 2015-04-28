#!/usr/bin/python

from year_prediction import *
import multiprocessing as mp

__author__ = 'snehas'

svr = mp.Process(target=regression, args=("svr",))
knn = mp.Process(target=regression, args=("knn",))
tree = mp.Process(target=regression, args=("tree",))
forest = mp.Process(target=regression, args=("forest",))

svr.start()
knn.start()
tree.start()
forest.start()