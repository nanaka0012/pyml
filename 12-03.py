import inspect
import numpy as np
import pandas

def list_size(variable, callers_local_vars):
    list_size_recursive(' # ', variable, callers_local_vars)

def list_size_recursive(header, variable, callers_local_vars):
    if type(variable) == list or type(variable) == pandas.core.series.Series:
        print(header, 'len(', str([k for k, v in callers_local_vars if v is variable][0]), ') = ', len(variable), sep='')
        if(len(variable) > 0):
            header += '  '
            list_size_recursive(header, variable[0], callers_local_vars)

def nparray_size(variable, callers_local_vars):
    if(type(variable) == np.ndarray):
        print(' # ', str([k for k, v in callers_local_vars if v is variable][0]), '.shape = ', variable.shape, sep='')

# shows the name and content of a given variable
def see(variable, all=0):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    print('# ', str([k for k, v in callers_local_vars if v is variable][0]), ' (', type(variable), ')', sep='', end='')
    if all or (type(variable) != list and type(variable) != np.ndarray and type(variable) != pandas.core.series.Series):
        print(' = ', variable, sep='')
    else:
        print('')
    list_size(variable, callers_local_vars)
    nparray_size(variable, callers_local_vars)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read iris data
attrNames = ["sepal length", "sepal width", "petal length", "petal width"]

dataList = []
classLabelsList = []
f = open('../tarotez/iris.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    if len(elems) > 1:
        dataList.append(elems[:-1])
        classLabelsList.append(elems[-1])
f.close()

x = np.float_(np.array(dataList))

# convert y to one hot
labels = np.unique(classLabelsList)

def label2onehot(label):
    return np.array([1 if label == elem else 0 for elem in labels])

y = np.array([label2onehot(label) for label in classLabelsList])

import numpy as np

# softmax function
def softmax(a):
    normalizer = np.sum(np.exp(a), axis=1)
    return np.array([exp_a / normalizer for exp_a in np.exp(a).transpose()]).transpose()

# a single-layer neural network with the softmax function
def softmax_nn(x_with_bias, W):
    return softmax(np.matmul(W, x_with_bias.transpose()).transpose())

# train a neural network
def train(x, y, learning_rate, iterNum):    
    x_with_bias = np.c_[x, np.ones(x.shape[0])]
    W = np.random.random((y.shape[1], x_with_bias.shape[1]))
    loss_dynamics = []
    accuracy_dynamics = []
    for _ in range(iterNum):
        loss = - np.sum(y * np.log(softmax_nn(x_with_bias, W)))
        loss_dynamics.append(loss)
        preds = np.argmax(softmax_nn(x_with_bias, W), axis=1)
        corrects = [1 if p == t else 0 for p, t in zip(preds, np.argmax(y, axis=1))]
        accuracy = np.sum(np.array(corrects)) / y.shape[0]
        accuracy_dynamics.append(accuracy)        
        W = update(x_with_bias, y, W, learning_rate)
    return W, loss_dynamics, accuracy_dynamics

learning_rate = 0.1
iterNum = 500
trained_W, loss_dynamics, accuracy_dynamics = train(x, y, learning_rate, iterNum) 

#update関数を自分で実装する必要があるらしい