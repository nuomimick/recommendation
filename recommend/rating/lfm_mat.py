from numpy.random import random
import numpy as np
from math import sqrt
import pandas as pd

# 以矩阵方式实现
class LFM:
    def __init__(self, lr, lamda, f, steps):
        self.__lr = lr
        self.__lamda = lamda
        self.__steps = steps
        self.__f = f

    def fit(self, train, test):
        train_x, train_y, test_x, test_y = train.values, train['rating'].values, test.values, test['rating'].values
        users = set(train_x[:, 0])
        items = set(train_x[:, 1])
        u_id = {k: v for k, v in zip(users, range(len(users)))}
        i_id = {k: v for k, v in zip(items, range(len(items)))}
        R = np.zeros((len(users), len(items)))

        self.P = random((self.__f, len(users))) / sqrt(self.__f)
        self.Q = random((self.__f, len(items))) / sqrt(self.__f)

        m = train_x.shape[0]
        for i in range(m):
            u, i, r = train_x[i][0], train_x[i][1], train_y[i]
            uidx, iidx = u_id[u], i_id[i]
            R[uidx, iidx] = r

        for iter in range(self.__steps):
            for i in range(m):
                u, i, r = train_x[i][0], train_x[i][1], train_y[i]
                uidx, iidx = u_id[u], i_id[i]
                err = r - np.dot(self.Q[:, iidx], self.P[:, uidx])
                temp = self.Q[:, iidx]
                self.Q[:, iidx] += self.__lr * (err * self.P[:, uidx] - self.__lamda * self.Q[:, iidx])
                self.P[:, uidx] += self.__lr * (err * temp - self.__lamda * self.P[:, uidx])

            m = test_x.shape[0]
            rmse = 0
            for i in range(m):
                u, i, r = test_x[i][0], test_x[i][1], test_y[i]
                uidx, iidx = u_id[u], i_id[i]
                rmse += (r - np.dot(self.Q[:, iidx], self.P[:, uidx])) ** 2
            print("rmse: %f" % sqrt(rmse / m))
        print('iteration finished')


if __name__ == '__main__':
    from recommend.data import datasets
    from sklearn.model_selection import train_test_split

    df = datasets.load_100k('pd').alldata
    df_train, df_test = train_test_split(df, test_size=0.2)
    uidset = set(df_test.user_id) - set(df_train.user_id)
    if uidset:
        df_test = df_test[~df_test["user_id"].isin(uidset)]
    iidset = set(df_test.item_id) - set(df_train.item_id)
    if iidset:
        df_test = df_test[~df_test["item_id"].isin(iidset)]

    lfm = LFM(0.01, 0.02, 10, 100)
    lfm.fit(df_train, df_test)
