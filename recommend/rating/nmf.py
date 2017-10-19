from numpy import ndarray
import numpy.random as nprd
import numpy as np
import pandas as pd

# 非负矩阵分解
class NMF:
    def __init__(self, f, reg_lambda, niters):
        self.f = f
        self.reg_lambda = reg_lambda
        self.niters = niters

        self.W = {}
        self.H = {}

    def transform(self, matrix ):
        uiMatrix = {}
        iuMatrix = {}
        if isinstance(matrix, ndarray):
            for line in matrix:
                u, i, r = line[:3]
                uiMatrix.setdefault(u, {})
                uiMatrix[u][i] = r
                iuMatrix.setdefault(i, {})
                iuMatrix[i][u] = r
        elif isinstance(matrix, pd.DataFrame):
            for idx,*line in matrix.itertuples():
                u, i, r = line[:3]
                uiMatrix.setdefault(u, {})
                uiMatrix[u][i] = r
                iuMatrix.setdefault(i, {})
                iuMatrix[i][u] = r
        return uiMatrix, iuMatrix

    def fit(self, ratingMatrix, testMatrix, fevals):
        lens = len(ratingMatrix)
        self.uiMatrix, self.iuMatrix = self.transform(ratingMatrix)
        test_uiMatrix, test_iuMatrix = self.transform(testMatrix)
        for u in self.uiMatrix:
            self.W.setdefault(u, nprd.random(self.f))
        for i in self.iuMatrix:
            self.H.setdefault(i, nprd.random(self.f))

        for iter in range(self.niters):
            for u in self.uiMatrix:
                tmp0, tmp1 = [], []
                for k in range(self.f):
                    tmp2, tmp3 = 0., 0.
                    for i in self.uiMatrix[u]:
                        tmp2 += self.uiMatrix[u][i] * self.H[i][k]
                        tmp3 += np.dot(self.W[u], self.H[i]) * self.H[i][k]
                    tmp0.append(tmp2)
                    tmp1.append(tmp3)
                self.W[u] *= np.array(tmp0) / np.array(tmp1)
            for i in self.iuMatrix:
                tmp0, tmp1 = [], []
                for k in range(self.f):
                    tmp2, tmp3 = 0., 0.
                    for u in self.iuMatrix[i]:
                        tmp2 += self.iuMatrix[i][u] * self.W[u][k]
                        tmp3 += np.dot(self.W[u], self.H[i]) * self.W[u][k]
                    tmp0.append(tmp2)
                    tmp1.append(tmp3)
                self.H[i] *= np.array(tmp0) / np.array(tmp1)

            results = []
            for u in test_uiMatrix:
                for i in test_uiMatrix[u]:
                    r = test_uiMatrix[u][i]
                    results.append((r - np.dot(self.W[u],self.H[i]))**2)
            rmse = np.sqrt(np.sum(results) / len(testMatrix))

            print("%d iter finish, rmse: %f" % (iter,rmse))

    def predict(self):
        pass


if __name__ == '__main__':
    from recommend.data import datasets
    from sklearn.model_selection import train_test_split
    mv = datasets.load_100k(type='pd')
    data = mv.alldata
    df_train, df_test = train_test_split(data,test_size=0.2,random_state=0)
    uidset = set(df_test.user_id) - set(df_train.user_id)
    if uidset:
        df_test = df_test[~df_test["user_id"].isin(uidset)]
    iidset = set(df_test.item_id) - set(df_train.item_id)
    if iidset:
        df_test = df_test[~df_test["item_id"].isin(iidset)]
    nmf = NMF(10,0.1,20)
    nmf.fit(df_train,df_test,"rmse")
