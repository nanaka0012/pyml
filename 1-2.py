import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(-10, 10, 100);
plt.figure(0)
plt.plot(x, x)
plt.show()

plt.plot(x, x*x)
plt.show()

plt.plot(x, math.e**-x)
plt.show()
