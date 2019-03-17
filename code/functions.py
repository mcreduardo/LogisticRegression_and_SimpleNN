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


def trainLogisticRegression(trainData, features):

    # initialize weights - [bias unit, input features]
    weights = initializeModel(features)

    # add bias unit to data
    ones = np.ones((trainData.shape[0], trainData.shape[1] +1))
    ones[:,1:] = ones[:,1:] * trainData
    trainData = ones



