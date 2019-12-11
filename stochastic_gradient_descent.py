# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_epochs = 50  # エポック数
m = 100  # データ数

t0, t1 = 5, 50


def learning_schedule(t):
    return t0/float(t+t1)


print(learning_schedule(3))

X = 2 * np.random.rand(m, 1)  # m*1の配列の要素を0から1のランダムに(の✕2)
y = 4 + 3 * X + np.random.randn(m, 1)  # 平均0、分散1（標準偏差1）m*1 正規分布

print('X : ', X)
print('y : ', y)

# format string in matplotlib : 'o'サークル、'b'青、'-'実線、'--'破線、'*'星 etc.
plt.plot(X, y, "*y")


# add x0 = 1 to each instance m*1の要素1の配列の右にXがくっつく
X_b = np.c_[np.ones((m, 1)), X]
print('X_b : ', X_b)
theta = np.random.randn(2, 1)  # 平均0、分散1（標準偏差1）2*1 正規分布

for epoch in range(n_epochs):  # エポック数まわす
    for i in range(m):  # データ数まわす
        # [0, m)の範囲で乱数、第2引数があれば[arg1, arg2)の範囲で乱数を返す
        random_index = np.random.randint(m)
        # 要素1つだけ取り出す、nparray[a, b]ならnparray[a]からnparray[b-1]まで
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        print('i : ', i, ', xi : ', xi, ', yi : ', yi, 'theta : ', theta)
        gradients = 2.0 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta * gradients

        if(epoch == 0 and i < 20):
            X_new = np.array([[0], [2]])
            X_new_b = np.c_[np.ones((2, 1)), X_new]
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, ":", color='lightblue')

plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, 2, 0, 15])
plt.savefig("stochastic_gradient_descent.png", formar='png', dpi=250)
# plt.show()
