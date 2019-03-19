######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# plot F1 score for both NN and LR
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
hiddenUnits = int(sys.argv[2])
max_epochs = int(sys.argv[3])
trainingSetPath = sys.argv[4]
testSetPath = sys.argv[5]

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
# Logistic Regression
for epochs in range(max_epochs):
    model = trainLogisticRegression\
        (trainingData, trainingClasses, encFeatures, epochs, learnRate, False)
    testLogisticRegression(trainingData, trainingClasses, model, False)
print("")
for epochs in range(max_epochs):
    model = trainLogisticRegression\
        (trainingData, trainingClasses, encFeatures, epochs, learnRate, False)
    testLogisticRegression(testData, testClasses, model, False)

print("")
print("")
######################################################################################
# NNet
for epochs in range(max_epochs):
    w_i_h, w_h_o = trainNNet\
        (trainingData, trainingClasses, encFeatures, epochs, learnRate, hiddenUnits, False)
    testNNet(trainingData, trainingClasses, w_i_h, w_h_o, False)
print("")
for epochs in range(max_epochs):
    w_i_h, w_h_o = trainNNet\
        (trainingData, trainingClasses, encFeatures, epochs, learnRate, hiddenUnits, False)
    testNNet(testData, testClasses, w_i_h, w_h_o, False)

print("")
print("")