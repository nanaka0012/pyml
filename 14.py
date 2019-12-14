#!/usr/bin/env python
# polynomial regression to abalone data

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

datalist = []
f = open('../tarotez/abalone.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    datalist.append(elems[1:])
f.close()
data = np.array(datalist)
x = np.c_[np.float_(data[:,0])]
y = np.c_[np.float_(data[:,4])]

lr = linear_model.LinearRegression()
lr2 = linear_model.LinearRegression()

lr2.fit(x, y)

predicted2 = lr2.predict(x)

X = np.hstack((x, np.power(x,2)))

lr.fit(X, y)

predicted = lr.coef_[0,1] * np.power(x,2) + lr.coef_[0,0] * x + lr.intercept_ * np.ones(x.shape)

error = np.array(y - predicted)
mse = (error ** 2).sum()

error2 = np.array(y - predicted2)
mse2 = (error2 ** 2).sum()

print(mse, mse2)

fig, ax = plt.subplots()
ax.scatter(x, y, c='blue', marker='o', label='observed', lw=0)
ax.scatter(x, predicted, c='red', marker='o', label='predicted', lw=0)
ax.scatter(x, predicted2, c='green', marker='o', label='predicted2', lw=0)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.legend(loc='upper left')
ax.set_xlabel('length', fontsize=18)
ax.set_ylabel('whole weight', fontsize=18)

plt.show()
