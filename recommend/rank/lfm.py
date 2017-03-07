from numpy.random import random
import numpy as np
import datasets
from sklearn.model_selection import train_test_split
import time

class LFM:
    def __init__(self,gamma,lamda,steps,f):
        self.__gamma = gamma
        self.__lamda = lamda
        self.__steps = steps
        self.__f = f
        self.__user_item = {}
        self.__item_user = {}
        self.p = {}
        self.q = {}

    def fit(self,train_x,train_y):
        m,n = train_x.shape
        for i in range(m):
            uid, iid, rating = train_x[i][0],train_x[i][1],train_y[i]
            self.__user_item.setdefault(uid,{})
            self.__user_item[uid][iid] = rating

            self.__item_user.setdefault(iid,{})
            self.__item_user[iid][uid] = rating

        for u in self.__user_item:
            self.p.setdefault(u,random(self.__f) / np.sqrt(self.__f))
        for i in self.__item_user:
            self.q.setdefault(i,random(self.__f) / np.sqrt(self.__f))

        t = time.time()
        for step in range(self.__steps):
            for u in self.__user_item:
                dict_items = self.__user_item[u]
                for i,r in dict_items.items():
                    e = r - np.dot(self.p[u],self.q[i])
                    tmp = self.q[i]
                    self.q[i] += self.__gamma * (e * self.p[u] - self.__lamda * tmp)
                    self.p[u] += self.__gamma * (e * tmp - self.__lamda * self.p[u])
        print(time.time() - t)
        print('iteration finished')

    def predict(self,test_x):
        return np.array([np.dot(self.p[u], self.q[i]) for u,i in test_x])

    def report(self,predict_y,test_y):
        length = len(test_y)
        mae = np.sum(abs(test_y - predict_y)) / length
        rmse = np.sqrt(np.sum(np.power((test_y - predict_y),2)) / length)
        print(mae,rmse)

if __name__ == '__main__':
    df = datasets.load_1m('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_deal(df,20,20,0.2)

    lfm = LFM(0.015, 0.02, 30, 50)
    lfm.fit(train_x,train_y) 
    lfm.report(lfm.predict(test_x),test_y)     