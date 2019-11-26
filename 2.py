#予測値の計算を内積でする
import numpy as np
import matplotlib.pyplot as plt

data = np.array(((20,30), (60,75), (80,30), (90,70), (95,90), (40,60), (80,90), (30,40), (25,55), (35,25), (80,45), (50,10), (25,80)))
y = np.array((0,1,0,1,1,1,1,0,0,0,1,0,1))
#w = (0.1, 0.3, -20)
w =  np.array((0.1, 0.3))
b = -20.0

# fig, ax = plt.subplots(figsize=(5,5))
# ax.axis('equal')
# ax.set_xlim((0, 100))
# ax.set_ylim((0, 100))
# ax.set_xlabel('Report score', size=20)
# ax.set_ylabel('Test score', size=20)
# plt.xticks(np.arange(0, 101, 20))
# plt.yticks(np.arange(0, 101, 20))
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(20)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(20)

# t = np.linspace(0,100,50)
# border_h = t
# border_v = - (t * w[0] + b) / w[1]

# ax.scatter(border_h, border_v, marker='.', c='black', s=2)
# ax.scatter(data[y==1,0], data[y==1,1], marker='*', c='blue')
# ax.scatter(data[y==0,0], data[y==0,1], marker='*', c='red')

# plt.show()

#新規データの入力
print("xとyの値")
test_x, test_y = (float(x) for x in input().split())
test_data = np.array((test_x, test_y))

#border_y = - (test_x * w[0] + w[2]) / w[1]
#difference = test_y - border_y
print(w)
print(test_data)
border_y = np.matmul(w, test_data)
print(border_y)


#合否判定
if border_y >= -b:
    print("1")
elif border_y < -b:
    print("0")