from numpy.random import random
import numpy as np

from sklearn.model_selection import train_test_split
import time

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
                item = self.items_pool[rd.randint(0,len(items_pool)-1)]
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



        for u in self.__user_item:
            self.p.setdefault(u,random(self.__f) / np.sqrt(self.__f))
        for i in self.__item_user:
            self.q.setdefault(i,random(self.__f) / np.sqrt(self.__f))

        for step in range(self.__steps):
            for u in self.__user_item:
                dict_items = self.__user_item[u]
                for i,r in dict_items.items():
                    e = r - np.dot(self.p[u],self.q[i])
                    tmp = self.q[i]
                    self.q[i] += self.__lr * (e * self.p[u] - self.__lamda * tmp)
                    self.p[u] += self.__lr * (e * tmp - self.__lamda * self.p[u])
            self.lr *= 0.9
        print('iteration finished')

    def predict(self,test_x):
        return np.array([np.dot(self.p[u], self.q[i]) for u,i in test_x])

    def report(self,predict_y,test_y):
        length = len(test_y)
        mae = np.sum(abs(test_y - predict_y)) / length
        rmse = np.sqrt(np.sum(np.power((test_y - predict_y),2)) / length)
        print("mae=%f,rmse=%f" % (mae, rmse))

if __name__ == '__main__':
    from recommend.data import datasets

    df = datasets.load_1m('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_deal(df,20,20,0.2)

    lfm = LFM(0.015, 0.02, 30, 50)
    lfm.fit(train_x,train_y) 
    lfm.report(lfm.predict(test_x),test_y)     