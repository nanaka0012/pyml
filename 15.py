import numpy as np

def linear_nn(x, w):
    y = np.matmul(x, w)
    return y