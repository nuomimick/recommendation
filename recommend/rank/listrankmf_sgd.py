import numpy as np
import pandas as pd
from math import exp, log
from numpy.random import random
import rank_metrics
from functools import reduce

# 使用sgd优化
class ListRankMF:
    '''
    classdocs
    '''
    tr_u2i = {}
    tr_i2u = {}
    te_u2i = {}
    U = {}
    V = {}

    def __init__(self, eta, fct, reg_lambda, steps):
        '''
        eta:学习率
        fd:因子维度
        reg_lambda:L2正则系数
        steps:迭代次数
        '''
        self.__eta = eta
        self.__lambda = reg_lambda
        self.__steps = steps
        self.__f = fct
        self.__user_item = {}
        self.__item_user = {}
        self.U = {}
        self.V = {}

    def fit(self, x_train, y_train, evals=None, top=None):
        m, n = x_train.shape
        for i in range(m):
            uid, iid, rating = x_train[i][0], x_train[i][1], y_train[i]
            self.__user_item.setdefault(uid, {})
            self.__user_item[uid][iid] = rating

            self.__item_user.setdefault(iid, {})
            self.__item_user[iid][uid] = rating

        for u in self.__user_item:
            self.U.setdefault(u, random(self.__f) / np.sqrt(self.__f))
        for i in self.__item_user:
            self.V.setdefault(i, random(self.__f) / np.sqrt(self.__f))

        exp_r = {}
        for u in self.__user_item:
            ratings_user = self.__user_item[u]
            exp_r[u] = sum([exp(r) for v, r in ratings_user.items()])

        for step in range(self.__steps):
            for u in self.__user_item:
                exp_uv = 0.
                for v, r in self.__user_item[u].items():
                    exp_uv += exp(self.__gfunc(u, v))
                last = None
                for v, r in self.__user_item[u].items():
                    new = exp(self.__gfunc(u, v))
                    old0, old1 = self.__gdfunc(u, v), self.U[u]
                    if not last is None:
                        exp_uv += new - last
                    self.U[u] -= self.__eta * ((new / exp_uv - 1) * exp(r) / exp_r[u] * self.__gdfunc(u, v) * self.V[
                        v] + self.__lambda * self.U[u])
                    self.V[v] -= self.__eta * (
                    (new / exp_uv - 1) * exp(r) / exp_r[u] * old0 * old1 + self.__lambda * self.V[v])
                    last = new
            print("第%d次迭代完成！" % (step + 1))
            loss = self.__loss()
            if evals and top:
                ndcg, precision, recall = self.evals(evals[0], evals[1], top)
            print(str(loss), str(ndcg), str(precision), str(recall))

    def __gfunc(self, i, j):
        x = sum(self.U[i] * self.V[j])
        return 5 / (1 + exp(-x))

    def __gdfunc(self, i, j):
        x = sum(self.U[i] * self.V[j])
        return 5 * exp(-x) / (1 + exp(-x)) ** 2

    def __loss(self):
        rst = 0.
        for u in self.__user_item:
            rst0, rst1 = 0., 0.
            for i, r in self.__user_item[u].items():
                rst0 += exp(r)
                rst1 += exp(self.__gfunc(u, i))
            rst2 = 0.
            for i, r in self.__user_item[u].items():
                rst2 += exp(r) / rst0 * log(exp(self.__gfunc(u, i)) / rst1)
            rst -= rst2
        U, V = [], []
        for row in self.U.items():
            U.append(row[1])
        for row in self.V.items():
            V.append(row[1])
        rst += 0.5 * self.__lambda * (np.linalg.norm(U) ** 2 + np.linalg.norm(V) ** 2)
        # print('loss:%f' % rst)
        return rst

    def rec_top(self, n):
        rec_dict = {}
        for u in self.__user_item:
            # 在不再训练集的所有物品中推荐
            rec_dict[u] = sorted(
                [(np.dot(self.U[u], self.V[v]), v) for v in self.__item_user if v not in self.__user_item[u]],
                reverse=True)[:n]
            rec_dict[u] = [t[1] for t in rec_dict[u]]
        return rec_dict

    def evals(self, x_test, y_test, top):
        rec_dict = self.rec_top(top)
        m, n = x_test.shape
        user_item = {}
        for i in range(m):
            u, i, r = x_test[i][0], x_test[i][1], y_test[i]
            user_item.setdefault(u, {})
            user_item[u][i] = r
        ndcg = []
        for u in rec_dict:
            temp = []
            for i in rec_dict[u]:
                if i in user_item[u]:
                    temp.append(user_item[u][i])
                else:
                    temp.append(0)
            ndcg.append(rank_metrics.ndcg_at_k(temp, top))
        # print('ndcg:%f' % np.mean(ndcg))

        p, r = 0., 0.
        for u in rec_dict:
            cm_users = set(user_item[u]) & set(rec_dict[u])
            p += len(cm_users) / top
            r += len(cm_users) / len(user_item[u])
        precision = p / len(rec_dict)
        recall = r / len(rec_dict)
        # print("precision=%f\nrecall=%f" % (precision, recall))\
        return np.mean(ndcg), precision, recall


def main():
    from recommend.data.datasets import load_100k
    import random
    # 数据集处理
    df = load_100k(type='pd').alldata
    nums_train = 10
    df = df.groupby('user_id').filter(lambda x: len(x) >= nums_train + 10)
    data_train = []
    data_test = []
    for gb, dtfm in df.groupby('user_id'):
        dtfm = dtfm.reset_index(drop=True)  # 去除原索引
        index_sample = random.sample(range(len(dtfm)), nums_train)
        data_train.append(dtfm.iloc[index_sample, :])
        data_test.append(dtfm.drop(index_sample))
    data_train = pd.concat(data_train)
    data_test = pd.concat(data_test)
    x_train = np.array(data_train[['user_id', 'item_id', 'timestamp']])
    y_train = np.array(data_train['rating'])
    x_test = np.array(data_test[['user_id', 'item_id', 'timestamp']])
    y_test = np.array(data_test['rating'])

    lrmf = ListRankMF(0.01, 10, 0.01, 800)
    lrmf.fit(x_train, y_train, evals=(x_test, y_test), top=10)


if __name__ == '__main__':
    main()
