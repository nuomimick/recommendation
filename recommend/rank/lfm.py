from numpy.random import random
import numpy as np
from random import randint
from operator import itemgetter

class LFM:
    def __init__(self,lr,lamda,steps,f):
        self.__lr = lr
        self.__lamda = lamda
        self.__steps = steps
        self.__f = f
        self.__user_item = {}
        self.__item_user = {}
        self.p = {}
        self.q = {}
        self.items_pool = []

    #negative sample
    def selectNegativeSample(self,n):
        for u in self.__user_item:
            items = self.__user_item[u].keys()
            length = len(items)
            count = 0
            for i in range(10 * length):
                item = self.items_pool[randint(0,len(self.items_pool)-1)]
                if item not in items:
                    self.__user_item[u][item] = 0
                    self.__item_user[item][u] = 0
                    count += 1
                if count >= (n * length):
                    break

    def fit(self,train_x,train_y):
        m,n = train_x.shape
        for i in range(m):
            uid, iid, rating = train_x[i][0],train_x[i][1],train_y[i]
            self.__user_item.setdefault(uid,{})
            self.__user_item[uid][iid] = 1

            self.__item_user.setdefault(iid,{})
            self.__item_user[iid][uid] = 1

            self.items_pool.append(iid)

        self.selectNegativeSample(1)

        for u in self.__user_item:
            self.p.setdefault(u,random(self.__f) / np.sqrt(self.__f))
        for i in self.__item_user:
            self.q.setdefault(i,random(self.__f) / np.sqrt(self.__f))

        for step in range(self.__steps):
            print(step)
            for u in self.__user_item:
                dict_items = self.__user_item[u]
                for i,r in dict_items.items():
                    e = r - np.dot(self.p[u],self.q[i])
                    tmp = self.q[i]
                    self.q[i] += self.__lr * (e * self.p[u] - self.__lamda * tmp)
                    self.p[u] += self.__lr * (e * tmp - self.__lamda * self.p[u])
            self.__lr *= 0.9
        print('iteration finished')

    def topN(self,test_x,top_n):
        all_items = set(self.__item_user)
        recommend_dict = {}
        for uid, iid in test_x:
            unrating_items = all_items - set(self.__user_item[uid])
            ratings = [(iid, np.dot(self.p[uid],self.q[iid])) for iid in unrating_items]
            ratings = sorted(ratings, key=itemgetter(1), reverse=True)[:top_n]
            recommend_dict.setdefault(uid, [tup[0] for tup in ratings])
        return recommend_dict

    def report(self,test_x,top_n=10):
        user_item = {}
        for u,i in test_x:
            user_item.setdefault(u,{})
            user_item[u][i] = 1
        recommend_dict = self.topN(test_x,top_n)
        p, r = 0., 0.
        for u in recommend_dict:
            cm_users = set(user_item[u]) & set(recommend_dict[u])
            p += len(cm_users) / top_n
            r += len(cm_users) / len(user_item[u])
        precision = p / len(recommend_dict)
        recall = r / len(recommend_dict)
        print("precision=%f,recall=%f" % (precision,recall))

if __name__ == '__main__':
    from recommend.data import datasets

    df = datasets.load_1m('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_deal(df,20,20,0.2)

    lfm = LFM(0.015, 0.02, 5, 50)
    lfm.fit(train_x,train_y) 
    lfm.report(test_x,10)