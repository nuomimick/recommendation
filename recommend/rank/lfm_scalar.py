from numpy.random import random as np_random
import numpy as np
from random import randint
from operator import itemgetter
import rank_metrics


class LFM:
    def __init__(self, lr, reg_lambda, steps, f):
        self.__lr = lr
        self.__lambda = reg_lambda
        self.__steps = steps
        self.__f = f
        self.__user_item = {}
        self.__item_user = {}
        self.p = {}
        self.q = {}
        self.items_pool = []

    def fit(self, train_x, train_y, evals, top):
        m, n = train_x.shape
        for i in range(m):
            uid, iid, rating = train_x[i][0], train_x[i][1], train_y[i]
            self.__user_item.setdefault(uid, {})
            self.__user_item[uid][iid] = rating

            self.__item_user.setdefault(iid, {})
            self.__item_user[iid][uid] = rating

            self.items_pool.append(iid)

        for u in self.__user_item:
            self.p.setdefault(u, np_random(self.__f) / np.sqrt(self.__f))
        for i in self.__item_user:
            self.q.setdefault(i, np_random(self.__f) / np.sqrt(self.__f))

        for step in range(self.__steps):
            for u in self.__user_item:
                dict_items = self.__user_item[u]
                for i, r in dict_items.items():
                    e = r - np.dot(self.p[u], self.q[i])
                    tmp = self.q[i]
                    self.q[i] += self.__lr * (e * self.p[u] - self.__lambda * tmp)
                    self.p[u] += self.__lr * (e * tmp - self.__lambda * self.p[u])
            self.__lr *= 0.9
            print("第%d次迭代完成！" % (step + 1))
            self.__loss()
            if evals and top:
                self.evals(evals[0], evals[1], top)

    def __loss(self):
        rst = 0.
        for u in self.__user_item:
            for i,r in self.__user_item[u].items():
                rst += (r - np.dot(self.p[u],self.q[i]))**2
        U, V = [], []
        for row in self.p.items():
            U.append(row[1])
        for row in self.q.items():
            V.append(row[1])
        rst += self.__lambda * (np.linalg.norm(U) ** 2 + np.linalg.norm(V) ** 2)
        print('loss:%f' % rst)

    def rec_top(self, n):
        rec_dict = {}
        for u in self.__user_item:
            # 在不再训练集的所有物品中推荐
            rec_dict[u] = sorted(
                [(np.dot(self.p[u], self.q[v]), v) for v in self.__item_user if v not in self.__user_item[u]],
                reverse=True)[:n]
            rec_dict[u] = [t[1] for t in rec_dict[u]]
        return rec_dict

    def evals(self, x_test, y_test, top_n):
        user_item = {}
        m, n = x_test.shape
        for i in range(m):
            u, i, r = x_test[i][0], x_test[i][1], y_test[i]
            user_item.setdefault(u, {})
            user_item[u][i] = r
        recommend_dict = self.rec_top(top_n)

        ndcg = []
        for u in recommend_dict:
            temp = []
            for i in recommend_dict[u]:
                if i in user_item[u]:
                    temp.append(user_item[u][i])
                else:
                    temp.append(0)
            ndcg.append(rank_metrics.ndcg_at_k(temp, top_n))
        print('ndcg:%f' % np.mean(ndcg))
        p, r = 0., 0.
        for u in recommend_dict:
            cm_users = set(user_item[u]) & set(recommend_dict[u])
            p += len(cm_users) / top_n
            r += len(cm_users) / len(user_item[u])
        precision = p / len(recommend_dict)
        recall = r / len(recommend_dict)
        print("precision=%f\nrecall=%f" % (precision, recall))


if __name__ == '__main__':
    from recommend.data.datasets import load_100k
    import random
    import pandas as pd

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

    lfm = LFM(0.001, 0.005, 30, 5)
    lfm.fit(x_train, y_train, evals=(x_test, y_test),top=10)
