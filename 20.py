import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def softmax(a):
    normalizer = np.sum(np.exp(a), axis=1)
    return np.array([exp_a / normalizer for exp_a in np.exp(a).transpose()]).transpose()

def softmax_nn(x_with_bias, W):
    return softmax(np.matmul(W, x_with_bias.transpose()).transpose())

def grad(x_with_bias, y, W):
    error = softmax_nn(x_with_bias, W) - y
    return np.matmul(error.transpose(), x_with_bias) / x_with_bias.shape[0]

def update(x_with_bias, y, W, learning_rate):
    g = grad(x_with_bias, y, W)
    delta_W = - learning_rate * g
    return W + delta_W

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

x_with_bias = np.c_[x, np.ones(x.shape[0])]
np.argmax(softmax_nn(x_with_bias, trained_W), axis=1)

plt.plot(loss_dynamics)
plt.show()
plt.plot(accuracy_dynamics)
plt.show()