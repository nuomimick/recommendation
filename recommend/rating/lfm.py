from numpy.random import random
import numpy as np
import pandas as pd


class LFM:
    def __init__(self, lr, lamda, f, steps):
        self.__lr = lr
        self.__lamda = lamda
        self.__steps = steps
        self.__f = f
        self.__user_item = {}
        self.__item_user = {}
        self.p = {}
        self.q = {}

    def transform(self, matrix):
        uiMatrix = {}
        iuMatrix = {}
        if isinstance(matrix, np.ndarray):
            for line in matrix:
                u, i, r = line[:3]
                uiMatrix.setdefault(u, {})
                uiMatrix[u][i] = r
                iuMatrix.setdefault(i, {})
                iuMatrix[i][u] = r
        elif isinstance(matrix, pd.DataFrame):
            for idx, *line in matrix.itertuples():
                u, i, r = line[:3]
                uiMatrix.setdefault(u, {})
                uiMatrix[u][i] = r
                iuMatrix.setdefault(i, {})
                iuMatrix[i][u] = r
        return uiMatrix, iuMatrix

    def fit(self, train, test):
        np.random.seed(0)
        # self.__user_item, self.__item_user = self.transform(train)
        # test_ui, test_iu = self.transform(test)

        self.__user_item, self.__item_user = train
        test_ui, test_iu = test

        for u in self.__user_item:
            self.p.setdefault(u, random(self.__f) / np.sqrt(self.__f))
        for i in self.__item_user:
            self.q.setdefault(i, random(self.__f) / np.sqrt(self.__f))

        for _ in range(self.__steps):
            for u in self.__user_item:
                dict_items = self.__user_item[u]
                for i, r in dict_items.items():
                    e = r - np.dot(self.p[u], self.q[i])
                    tmp = self.q[i]
                    self.q[i] += self.__lr * (e * self.p[u] - self.__lamda * tmp)
                    self.p[u] += self.__lr * (e * tmp - self.__lamda * self.p[u])

            # 另一种更新方式
            # for u in self.__user_item:
            #     temp = 0
            #     for i in self.__user_item[u]:
            #         r = self.__user_item[u][i]
            #         e = r - np.dot(self.p[u],self.q[i])
            #         temp += (e * self.q[i] - self.__lamda * self.p[u])
            #     self.p[u] += self.__lr * temp
            #
            # for i in self.__item_user:
            #     temp = 0
            #     for u in self.__item_user[i]:
            #         r = self.__item_user[i][u]
            #         e = r - np.dot(self.p[u],self.q[i])
            #         temp += (e * self.p[u] - self.__lamda * self.q[i])
            #     self.q[i] += self.__lr * temp


            rmse, count = 0, 0
            for u in test_ui:
                for i in test_ui[u]:
                    r = test_ui[u][i]
                    rmse += (np.dot(self.p[u], self.q[i]) - r) ** 2
                    count += 1
            print("%d iter,rmse: %f" % (_,np.sqrt(rmse / count)))

        print('iteration finished')

    def predict(self, test_x):
        return np.array([np.dot(self.p[u], self.q[i]) for u, i in test_x])

    def report(self, predict_y, test_y):
        length = len(test_y)
        mae = np.sum(abs(test_y - predict_y)) / length
        rmse = np.sqrt(np.sum(np.power((test_y - predict_y), 2)) / length)
        print(mae, rmse)


if __name__ == '__main__':
    from recommend.data import datasets
    from sklearn.model_selection import train_test_split

    # df = datasets.load_100k('pd').alldata
    # df_train, df_test = train_test_split(df, test_size=0.2)
    # uidset = set(df_test.user_id) - set(df_train.user_id)
    # if uidset:
    #     df_test = df_test[~df_test["user_id"].isin(uidset)]
    # iidset = set(df_test.item_id) - set(df_train.item_id)
    # if iidset:
    #     df_test = df_test[~df_test["item_id"].isin(iidset)]

    df_train, df_test = datasets.split_musical_instruments()
    lfm = LFM(0.01, 0.05, 10, 100)
    lfm.fit(df_train[:-1], df_test[:-1])
