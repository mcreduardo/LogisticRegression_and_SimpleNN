######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# Logistic Regression using NN structure and stochastic gradient descent
######################################################################################

import json
import numpy as np 
import sys

# Import Functions
from functions import *

# make sure your is deterministic
np.random.seed(0)

######################################################################################
# Get inputs and load .json data

# Receive arguments using sys
learnRate = float(sys.argv[1])
epochs = int(sys.argv[2])
trainingSetPath = sys.argv[3]
testSetPath = sys.argv[4]

# Load training and test set
# metadata + data
with open(trainingSetPath) as f:
    trainSet = json.load(f)
with open(testSetPath) as f:
    testSet = json.load(f)

# Extract features and classes from metadata
features = trainSet["metadata"]["features"][:-1]
numberFeatures = len(features)

# Discard Metadata
trainingData = trainSet["data"]
testData = testSet["data"]

######################################################################################
# Prepare Data: one hot encoding and standarization

positiveClass = trainSet["metadata"]["features"][-1][1][1]
# see functions file for details on one hot encoding
# training data
trainingData, encFeatures, trainingClasses =\
    oneHotEncoding(trainingData, features, positiveClass)
# test data
testData, _, testClasses =\
    oneHotEncoding(testData, features, positiveClass)

# Standaize numeric features
# https://www.biostat.wisc.edu/~craven/cs760/hw/standardizing.pdf
mean, stddev = getMeanStddevData(trainingData, encFeatures)
trainingData = standarize(trainingData, mean, stddev)
testData = standarize(testData, mean, stddev)

######################################################################################
# Training

trainLogisticRegression(trainingData, encFeatures)

######################################################################################
# Testing


