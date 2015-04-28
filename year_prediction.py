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
from sklearn import decomposition

__author__ = 'snehas'


TRAIN_DATA = "/Users/snehas/data/msd_dataset/train_sample.npy"
TEST_DATA = "/Users/snehas/data/msd_dataset/test_sample.npy"

#Get the hyperparameters of the Support vector regression
def get_hyperparams_svr(x_data, y_class):
    scores = dict()

    #Specify the values being searched over
    kernel_list = ['rbf']
    gamma_list = [0.0001, 0.001, 0.001, 0, 0.01]
    epsilon_list = [0, 0.01, 0.1, 0.2, 0.5]
    C_list = [1,0.2,0.3,0.4,0.5]

    #Variables to store the final hyperparameters
    min_score = 9999.00
    max_kernel = ""
    max_gamma = -1.0
    max_epsilon = -1.0
    max_C = -1.0

    #For every set of hyperparameters, perform cross-validation using MAE as scoring function
    for kernel in kernel_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                for C in C_list:
                    clf = SVR(kernel=kernel, C = C, gamma = gamma, epsilon=epsilon)
                    print "Doing " + kernel + ", " + str(gamma) + ", " + str(C) + ", " + str(epsilon)
                    score = -cross_val_score(clf, x_data, y_class, cv = 5, scoring='mean_absolute_error').mean()
                    scores[kernel,gamma,epsilon,C] = score
                    #Get least error
                    if(score < min_score):
                        min_score = score
                        max_kernel = kernel
                        max_epsilon = epsilon
                        max_C = C
                        max_gamma = gamma

    for kernel in kernel_list:
        for gamma in gamma_list:
            for epsilon in epsilon_list:
                for C in C_list:
                    print kernel + " " + str(gamma) + " " + str(epsilon) + " " + str(C) + " " + str(scores[kernel,gamma,epsilon,C])

    #Printing the output
    print max_C, max_gamma, max_kernel, max_epsilon
    return max_C, max_gamma, max_kernel, max_epsilon

#Function to spit the output in an appropriately named CSV file
def output(data):
    with open("output.csv", "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID', 'Target'])
        for i in range(0, len(data)):
            writer.writerow(["{0:0.1f}".format(float(i+1)), "{0:0.1f}".format(data[i])])

def compute_rmse(pred, actual):
    return np.sqrt(((pred - actual) ** 2).mean())

def compute_confusion_matrix(pred, actual):
    #change to years
    low = 1922.0
    high = 2011.0
    for i in range(0, len(pred)):
        pred[i] = int(pred[i]*(high-low) + low)
        actual[i] = int(actual[i]*(high-low) + low)
    pred_hist =  np.histogram(pred, bins = range(1920,2020,10))
    actual_hist = np.histogram(actual, bins = range(1920,2020,10))

    print pred_hist
    print actual_hist





def compute_mae(pred, actual):
    sum = 0.0
    for i in range(0, len(pred)):
        sum = sum + abs(pred[i] - actual[i])
    sum = sum/len(pred)
    return sum

#Backward stepwise selection
def backwardStepwiseSelection(regress, data, y_class):
    print "Backward stepwise selection"
    min_score = 99999
    max_k = []
    scores = dict()

    #Run for k = p-1 to 1; p = number of predictors
    for k in range(len(data[0]) -1, 1, -1):
        min_step_score = 99999
        step_data = np.delete(data, k, 1)
        score = -cross_val_score(regress, step_data, y_class, scoring='mean_squared_error').mean()
        if(score < min_step_score):
            min_step_score = score
            max_k = [k]
        #Run from 0 to k, delete each feature and see which has the least MAE
        for i in range(0, k):
            print "Doing: " + str(k)
            x_data = np.delete(step_data, i, 1)
            score = -cross_val_score(regress, x_data, y_class, scoring='mean_squared_error').mean()
            #Get least error
            if(score < min_step_score):
                min_step_score = score
                max_k = [k,i]
        scores[k] = min_step_score
        if(min_step_score < min_score):
            min_score = min_step_score

    for k in range(len(data[0]) -1, 1, -1):
        print str(k) + " " + str(scores[k])

    print "Max k, backward stepwise selection: " + str(max_k)
    #Return features to be deleted
    return max_k


def regression(method, train_location=TRAIN_DATA, test_location=TEST_DATA, verbose=False):
    train = np.load(train_location)

    #Get the required training data
    x_data = train[:, 1:]  #All but the first column
    y_class = train[:, 0]

    print "Loading data: " + method
    #Get the test data
    test = np.load(test_location)
    test_data = test[:, 1:]

    print "Do PCA"
    #Do PCA
    pca = decomposition.KernelPCA()
    x_data = pca.fit_transform(x_data)
    test_data = pca.transform(test_data)



    if(method == "svr"):
        #C, gamma, kernel, epsilon = get_hyperparams_svr(x_data, y_class)
        #regress = SVR(kernel=kernel, C = C, gamma = gamma, epsilon=epsilon)
        #parameters = {'kernel':['rbf'], 'C':[1,0.2,0.3,0.4,0.5], 'gamma': [0.0001, 0.001, 0.001, 0, 0.01], 'epsilon' :[0, 0.01, 0.1, 0.2, 0.5]}
        regress = svr = SVR(C=0.2,gamma= 0.01,kernel='rbf',epsilon=0.01)
        #regress = grid_search.GridSearchCV(svr, parameters)
    elif(method == "tree"):
        regress = DecisionTreeRegressor()
    elif(method == "forest"):
        regress = ensemble.RandomForestRegressor()
    elif (method == "knn"):
        regress = neighbors.KNeighborsRegressor()
    elif (method == "lin"):
        regress = RidgeCV()
        #max_k = backwardStepwiseSelection(regress, x_data, y_class)
        #x_data = np.delete(x_data, max_k, 1)
        #test_data = np.delete(test_data, max_k, 1)
    elif(method == "sgd"):
        regress = SGDRegressor()


    print "Training " + method
    regress.fit(x_data, y_class)

    print "Prediction " + method
    out = regress.predict(test_data)
    #output(out)

    rmse = compute_rmse(out, test[:, 0])
    print "RMSE: " + method + " :" + str(rmse)

    compute_confusion_matrix(out, test[:, 0])

    mae = compute_mae(out, test[:,0])
    #print "MAE: " + method + " :" + str(mae)

    fp = open("output_" + method + ".txt","w")
    fp.write("RMSE: " + method + " :" + str(rmse))
    fp.write("MAE: " + method + " :" + str(mae))
    fp.close()




if __name__ == "__main__":
    t1 = time()
    regression(sys.argv[1])
    t2 = time()
    print 'Time taken in seconds: %f' % (t2 - t1)