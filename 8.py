#行列 W = np.array([[5,3],[2,-1],[4,2]])とベクトル x = np.array([3,2])の積をnp.matmul によって求め、それが手計算で行った結果と一致することを確認せよ。

import numpy as np

W = np.array(
    [[5,3],
    [2,-1],
    [4,2]])
x = np.array([3,2])

print(np.matmul(W, x))