import numpy.random as nprd
import numpy as np
from time import time
from math import sqrt

# k1对应用户，k2对应物品
class TopicMF:
    # 使用梯度下降时的参数
    def __init__(self, lr, reg_lambda0, reg_lambda1, f, niters):
        self.lr = lr
        self.reg_lambda0 = reg_lambda0  # 评分函数正则化系数
        self.reg_lambda1 = reg_lambda1  # 评论函数权重
        self.f = f
        self.niters = niters

        self.U = {}
        self.V = {}
        self.theta = {}
        self.phi = {}
        self.k1 = 1
        self.k2 = 1

    def computeTheta(self, i, j):
        ui = self.U[i]
        ij = self.V[j]
        dividend = np.exp(self.k1 * np.abs(ui) + self.k2 * np.abs(ij))
        divisor = np.sum(dividend)
        return dividend / divisor

    def fit(self, trainMatrix, testMatrix, fevals):
        print("begin initation！")
        ratingMatrixUI, ratingMatrixIU, reviewMatrix = trainMatrix
        test_ratingMatrixUI, test_ratingMatrixIU, test_reviewMatrix = testMatrix
        for u in ratingMatrixUI:
            self.U.setdefault(u, nprd.random(self.f) / sqrt(self.f))
        for i in ratingMatrixIU:
            self.V.setdefault(i, nprd.random(self.f) / sqrt(self.f))
        words = {k for ct in reviewMatrix.values() for k, v in ct.items()}
        for w in words:
            self.phi.setdefault(w, nprd.random(self.f) / sqrt(self.f))
        for u in ratingMatrixUI:
            for i in ratingMatrixIU:
                self.theta.setdefault((u, i), self.computeTheta(u, i))
        # for d in reviewMatrix:
        #     for w in reviewMatrix[d]:
        #         reviewMatrix[d][w] /= len(words)
        print("end initation")
        print("begin training")
        for iter in range(self.niters):
            time_begin = time()
            dict_v = {}
            tp0, tp1 = 0, 0
            for u in ratingMatrixUI:
                tmp0, tmp1 = 0., 0.
                abs_u, sign_u = np.abs(self.U[u]), np.sign(self.U[u])
                for i in ratingMatrixUI[u]:
                    r = ratingMatrixUI[u][i]
                    tmp0 += self.V[i] * (np.dot(self.U[u], self.V[i]) - r) + self.reg_lambda0 * self.U[u]
                    theta = self.theta[(u, i)]
                    abs_v, sign_v = np.abs(self.V[i]), np.sign(self.V[i])
                    temp = theta * (1 - theta)
                    temp_u, temp_v = temp * sign_u, temp * sign_v
                    temp_absu, temp_absv = abs_u - np.dot(theta, abs_u), abs_v - np.dot(theta, abs_v)
                    temp0, temp1 = 0, 0
                    for w in reviewMatrix[(u, i)]:
                        freq = reviewMatrix[(u, i)][w]
                        err = np.dot(theta, self.phi[w]) - freq
                        temp0 += self.k1 * err * temp_u * self.phi[w]
                        temp1 += self.k2 * err * temp_v * self.phi[w]

                        # k1,k2
                        theta_p = theta * self.phi[w]
                        tp0 += np.dot(err * theta_p, temp_absu)
                        tp1 += np.dot(err * theta_p, temp_absv)
                    dict_v.setdefault((u, i), temp1)
                    tmp1 += temp0
                self.U[u] -= self.lr * (tmp0 + self.reg_lambda1 * tmp1)

            for i in ratingMatrixIU:
                tmp0, tmp1 = 0., 0.
                for u in ratingMatrixIU[i]:
                    r = ratingMatrixIU[i][u]
                    tmp0 += self.U[u] * (np.dot(self.U[u], self.V[i]) - r) + self.reg_lambda0 * self.V[i]
                    tmp1 += dict_v[(u, i)]
                self.V[i] -= self.lr * (tmp0 + self.reg_lambda1 * tmp1)

            self.k1 -= self.lr * self.reg_lambda1 * tp0
            self.k2 -= self.lr * self.reg_lambda1 * tp1

            for u in ratingMatrixUI:
                for i in ratingMatrixIU:
                    self.theta[(u, i)] = self.computeTheta(u, i)

            for w in words:
                ks1,ks2 = [], []
                for d in reviewMatrix:
                    ks1.append(reviewMatrix[d].get(w, 0) * self.theta[d])
                    ks2.append(np.dot(self.theta[d], self.phi[w]) * self.theta[d])
                self.phi[w] *= np.array(ks1).sum(axis=1)/np.array(ks2).sum(axis=1)

            rmse, counts = 0., 0
            for u in test_ratingMatrixUI:
                for i in test_ratingMatrixUI[u]:
                    r = test_ratingMatrixUI[u][i]
                    rmse += (np.dot(self.U[u], self.V[i]) - r) ** 2
                    counts += 1
            rmse = sqrt(rmse / counts)

            time_end = time()
            print("%d iter,time all costs: %f,rmse: %f" % (iter, time_end - time_begin, rmse))

    def predict(self):
        pass


if __name__ == '__main__':
    from recommend.data import datasets

    train, test = datasets.split_musical_instruments()
    topicmf = TopicMF(0.05, 0.05, 0.05, 10, 40)
    topicmf.fit(train, test, "rmse")
