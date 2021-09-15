import numpy as np
import sklearn.datasets as datasets  # 数据集模块
from sklearn.model_selection import train_test_split  # 划分训练集和验证集
import matplotlib.pyplot as plt
import random
from KNN import KNN
# 读取数据集
x, y = datasets.load_digits(return_X_y=True)



# 划分训练集和验证集,使用sklearn中的方法
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

# KNN最近邻进行分类
k = 7
knn = KNN(x_test, x_train, k)
pred = knn.knn(x_test, x_train, y_train)

# 分类准确率
accuracy = np.mean(pred == y_test)
print(accuracy)
