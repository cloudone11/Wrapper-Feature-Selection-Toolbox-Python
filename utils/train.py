import time
import numpy as np
import matplotlib.pyplot as plt

# from knn import *
from gpu_knn import *

# data generation
np.random.seed(272)
data_size_1 = 3000
x1_1 = np.random.normal(loc=10.0, scale=2.0, size=data_size_1)
x2_1 = np.random.normal(loc=8.0, scale=2.0, size=data_size_1)
# y1_1 = [0 for i in range(data_size_1)]
y1_2 = np.zeros(data_size_1, dtype=np.int32)

data_size_2 = 4000
x1_2 = np.random.normal(loc=20.0, scale=4.0, size=data_size_2)
x2_2 = np.random.normal(loc=16.0, scale=4.0, size=data_size_2)
# y2_1 = [1 for i in range(data_size_2)]
y2_2 = np.ones(data_size_2, dtype=np.int32)

# 将数组拼接起来
x1 = np.concatenate((x1_1, x1_2), axis=0)
x2 = np.concatenate((x2_1, x2_2), axis=0)
# 沿着水平方向将数组堆叠起来. 传入是以元组的形式, 返回ndarray对象
x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
y = np.concatenate((y1_2, y2_2), axis=0)

# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()

data_size_all = data_size_1 + data_size_2
# 随机置换序列
shuffled_index = np.random.permutation(data_size_all)
# 打乱顺序
x = x[shuffled_index]
y = y[shuffled_index]

split_index = int(data_size_all * 0.7)
x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]

# visualize data
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
# plt.show()
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='.')
# plt.show()

# visualize data
# plt.scatter(x_train[:, 0], x_train[:, 1], marker='.')
# plt.show()
# plt.scatter(x_test[:, 0], x_test[:, 1], marker='.')
# plt.show()

x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))

time_start = time.time()
clf = KNN(k=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('test accuracy: {:.3}'.format(clf.score(y_test, y_pred)))
time_end = time.time()
print('tiem : %f'%(time_end - time_start))

