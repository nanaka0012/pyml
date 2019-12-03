#ReLU関数の実装
#入力した値が0以下のとき0になり、1より大きいとき入力をそのまま出力する

import numpy as np
import matplotlib.pylab as plt

def relu(x):
  return np.maximum(0, x)

print("値の入力")
x = int(input())
print(relu(x))