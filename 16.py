import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
d = pd.read_csv('../tarotez/GlobalTemperatures.csv')
offset = 100
date = d['dt'][offset::12]
year = np.array([int(e.split('-')[0]) for e in date])
temperature = d['LandAverageTemperature'][offset::12]
lr = linear_model.LinearRegression()
lr.fit(year.reshape((-1,1)), temperature.values)
year_test = np.linspace(year.min(), year.max(), len(year))
temperature_predict = lr.predict(year_test.reshape((-1,1)))

def linear_nn(x, w):
    y = np.matmul(x, w)
    return y

ones = np.ones_like(year)
x = np.hstack((year.reshape(len(year), 1), ones.reshape(len(ones), 1)))
w = (1.0/500, 7.2)
print(x)

y = linear_nn(x, w)

error = np.array(temperature - y)
mse = (error ** 2).mean()

error2 = np.array(temperature - temperature_predict)
mse2 = (error2 ** 2).mean()

print(mse, mse2)

fig, ax = plt.subplots()
ax.scatter(year, temperature, c='blue', marker='o', label='observed', lw=0)
ax.scatter(year, y, c='red', marker='o', label='predicted', lw=0)
ax.scatter(year, temperature_predict, c='green', marker='o', label='predicted2', lw=0)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.legend(loc='upper left')
ax.set_xlabel('length', fontsize=18)
ax.set_ylabel('whole weight', fontsize=18)

plt.show()