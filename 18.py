import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def linear_nn(x, w):
    y = softmax(np.matmul(x, w))
    return y

def differential_cross_entropy(w, x, t):
    return (linear_nn(x, w) - t) * x