######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# Logistic Regression using NN structure and stochstic gradient descent
# 
# Aux file with functions
######################################################################################

import numpy as np 

######################################################################################

# Apply Sigmoid Function
def sigmoid(X):
    # overflow for large numbers
    return 1 / (1 + np.e**(-X))

# compute cost
def computeCost(predicted, actual):
    y = actual
    y_hat = predicted
    return np.sum(- y * np.log(y_hat) - (1-y) * np.log(1-y_hat))
    

######################################################################################
# One hot encoding for categorical feature
def oneHotEncoding(data, features, positiveClass):

    # size of features vector
    # new features structure
    # 'numeric' for numeric
    # feature = ['a', 'b', 'c'] -->> 3 new features 'a', ...
    # Ex.: [[‘feature1’, ‘numeric’],
    #      [‘feature2’, [‘cat’, ‘dog’, ‘fish’]]
    #      --->>> [‘numeric', ‘cat’, ‘dog’, ‘fish’]
    newFeatures = []
    for feature in features:
        if feature[1] == "numeric": 
            newFeatures.append('numeric')
        else: 
            for string in feature[1]: newFeatures.append(string)

    # numpy array for new data
    encodedData = np.zeros((len(data), len(newFeatures)))
    # list for classes
    classes = []

    # one hot encoding  
    for i in range(len(data)):
        k = 0 # idx for old features vector
        j = 0 # idx for new features vector
        while j in range(len(newFeatures)):
            if newFeatures[j] == 'numeric':
                encodedData[i][j] = data[i][k]
                k += 1
                j += 1
            else:
                encodedData[i][j + features[k][1].index(data[i][k])] = 1
                j += len(features[k][1])
                k += 1
        classes.append(data[i][-1] == positiveClass)

    # return encoded data, new features vector, classes
    return encodedData, newFeatures, classes



######################################################################################
# Standarization
# Functions:
# 1) Compute mean and stddev
# 2) apply to dataset
# https://www.biostat.wisc.edu/~craven/cs760/hw/standardizing.pdf

def getMeanStddevData(data, features):

    mean = np.sum(data, axis = 0)/len(data)
    stddev_sum = np.sum((data - np.tile(mean,(len(data),1)) )**2, axis = 0)
    stddev = np.sqrt(stddev_sum/len(data))

    # avoid division by 0 - convetion:
    # also, if categorical feature, dont standarize
    for i in range(len(stddev)):
	    if stddev[i] == 0 or features[i] != 'numeric': 
		    stddev[i] = 1 
		    mean[i]  = 0

    return mean, stddev

def standarize(data, mean, stddev):
    # (x - mean)/stddev
    return np.divide(data - np.tile(mean,(len(data),1)) , \
        np.tile(stddev,(len(data),1)))  

######################################################################################
# Training - Logistic regression

def initializeModel(features):
    # weights ->>> input features + bias unit
    sizeModel = len(features)+1
    weights = np.random.uniform(low=-0.01, high=0.01, size=(1, sizeModel))
    return weights


def trainLogisticRegression(trainData, classes, features, epochs, learnRate):

    # initialize weights - [bias unit, input features]
    weights = initializeModel(features)

    # add bias unit to data
    ones = np.ones((trainData.shape[0], trainData.shape[1] +1))
    ones[:,1:] = ones[:,1:] * trainData
    trainData = ones

    # train
    for j in range(epochs):
        loss = 0
        correct = 0
        misclassified = 0
        for i in range(len(trainData)):
            # predict
            output = sigmoid(np.sum(weights * trainData[i]))
            if round(output) == classes[i]: correct += 1
            else: misclassified += 1
            # loss
            loss += computeCost(output, classes[i])
            # correct
            weights = weights + learnRate * (classes[i]-output) * trainData[i]

        # print training output
        print(j+1, end = " ")
        print("%.12f"%loss, end = " ")
        print(correct, end = " ")
        print(misclassified)

    return weights


######################################################################################
# Testing - Logistic regression

def testLogisticRegression(testData, classes, weights):

    # add bias unit to data
    ones = np.ones((testData.shape[0], testData.shape[1] +1))
    ones[:,1:] = ones[:,1:] * testData
    testData = ones

    # var to calculate precision recall
    truePositive = 0
    falseNegative = 0
    falsePositive = 0

    correct = 0
    misclassified = 0
    for i in range(len(testData)):
        # predict
        output = sigmoid(np.sum(weights * testData[i]))
        if round(output) == classes[i]: correct += 1
        else: misclassified += 1

        # precision recall
        if classes[i] == 1:
            if round(output) == 1: truePositive += 1
            else:  falseNegative += 1
        else: 
            if round(output) == 1: falsePositive += 1

        # print instance output
        print("%.12f"%output, end = " ")
        print(int(round(output)), end = " ")
        print(int(classes[i]))

    # print set output
    print(correct, end = " ")
    print(misclassified)

    precision = truePositive/(truePositive + falsePositive)   
    recall = truePositive/(truePositive + falseNegative)
    F1_score = 2 * precision * recall / (precision + recall)
    print("%.12f"%F1_score)



######################################################################################
# Training - 1 hidden layer NN

def initializeModel(features, hiddenUnits):
    # weights ->>> input features + bias unit
    numberInputs = len(features)+1
    w_i_h = np.random.uniform(low=-0.01, high=0.01, size=(hiddenUnits, numberInputs))
    w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, hiddenUnits + 1))
    return w_i_h, w_h_o


def trainNNet(trainData, classes, features, epochs, learnRate, hiddenUnits):

    # initialize weights
    # jth input unit to the ith hidden unit
    # ith hidden unit to the output unit
    w_i_h, w_h_o = initializeModel(features, hiddenUnits)

    # add bias unit to data
    ones = np.ones((trainData.shape[0], trainData.shape[1] +1))
    ones[:,1:] = ones[:,1:] * trainData
    trainData = ones

    # train
    for j in range(epochs):
        loss = 0
        correct = 0
        misclassified = 0
        for i in range(len(trainData)):
            # predict hidden layer output
            hiddenLayerOutput = sigmoid(np.matmul(w_i_h, trainData[i]))
            # add bias unit
            ones = np.ones(len(hiddenLayerOutput) + 1)
            ones[1:] *= hiddenLayerOutput
            hiddenLayerOutput = ones
            # predict output
            output = sigmoid(np.sum(w_h_o * hiddenLayerOutput))
            if round(output) == classes[i]: correct += 1
            else: misclassified += 1
            # loss
            loss += computeCost(output, classes[i])
            # correct
            errorOutput = classes[i]-output
            errorHiddenUnits =\
                hiddenLayerOutput * (1 - hiddenLayerOutput) * errorOutput * w_h_o[0]
            w_h_o = w_h_o + learnRate * errorOutput * hiddenLayerOutput
            w_i_h = w_i_h + learnRate * np.outer(errorHiddenUnits[1:], trainData[i])
        # print training output
        print(j+1, end = " ")
        print("%.12f"%loss, end = " ")
        print(correct, end = " ")
        print(misclassified)

    #return weights