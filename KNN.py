import numpy as np


class KNN():
    def __init__(self, x_test, x_train, k):
        self.neighbor_distance = np.zeros((len(x_test), len(x_train)))
        self.pred = np.zeros(len(x_test,))
        self.neighbors = k

    def knn(self, x_test, x_train, y_train):
        for i in range(len(x_test)):
            for j in range(len(x_train)):
                self.neighbor_distance[i][j] = self.compute_distance(
                    x_test[i], x_train[j])
            self.pred[i] = self.compute_pred(
                self.neighbor_distance[i], y_train)
        return self.pred

    def compute_distance(self, x, y):
        # 这里使用mae来进行计算距离
        distance = np.sum(np.abs(x - y))
        # distance = np.sqrt(np.square(x-y)/len(x))
        # distance = np.sum(distance)
        return distance

    def compute_pred(self, distance, y_train):
        k_pred_index = distance.argsort()[:self.neighbors]
        k_pred = [y_train[index] for index in k_pred_index]
        pred = np.argmax(np.bincount(k_pred))
        return pred
