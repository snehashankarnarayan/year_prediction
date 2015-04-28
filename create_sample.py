#!/usr/bin/python
__author__ = 'snehas'

import numpy as np
import random

TRAIN_DATA = "/Users/snehas/data/msd_dataset/train.npy"
TEST_DATA = "/Users/snehas/data/msd_dataset/test.npy"

train = np.load(TRAIN_DATA)
test = np.load(TEST_DATA)

len_train = len(train)
len_test = len(test)

num_train = 10000
num_test = 1000

#train_indices = np.random_integers(0, len_train-1, num_train)
#test_indices = np.random_integers(0, len_test-1, num_test)

train_sample = random.sample(train, num_train)
test_sample = random.sample(test, num_test)

np.save("train_sample.npy", train_sample)
np.save("test_sample.npy", test_sample)