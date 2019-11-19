import numpy as np

A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])

print(3*A)
print(np.dot(A, B))
print(A*B)
print(np.dot(B, A))
