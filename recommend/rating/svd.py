import numpy as np
import pandas as pd
from numpy.random import random
import math

class SVD:
    '''
    classdocs
    '''
    def __init__(self,lr,lamda,ft,steps):
        '''
        lr: learing rate
        lamda: egularization factor
        ft:factor dimension
        '''
        self.__lr = lr
        self.__lamda = lamda
        self.__ft = ft
        self.__steps = steps
        self.__user_item = {}
        self.__item_user = {}
        self.__pu = {}
        self.__qi = {}
        self.__bu = {}
        self.__bi = {}
        self.mean = 0.

    def fit(self, train_x, train_y):
        self.mean = np.mean(train_y)
        m, n = train_x.shape
        for i in range(m):
            uid, iid, rating = train_x[i][0],train_x[i][1],train_y[i]
            self.__user_item.setdefault(uid,{})
            self.__user_item[uid][iid] = rating

            self.__item_user.setdefault(iid,{})
            self.__item_user[iid][uid] = rating

        for u in self.__user_item:
            self.__bu.setdefault(u,0)
            self.__pu.setdefault(u,random(self.__ft) / math.sqrt(self.__ft))
        for i in self.__item_user:
            self.__bi.setdefault(i,0)
            self.__qi.setdefault(i,random(self.__ft) / math.sqrt(self.__ft))

        for _ in range(self.__steps):
            for u in self.__user_item:
                items = self.__user_item[u]
                for i,r in items.items():
                    err = r - self.__rating(u,i)
                    self.__bu[u] += self.__lr * (err - self.__lamda * self.__bu[u])
                    self.__bi[i] += self.__lr * (err - self.__lamda * self.__bi[i])
                    temp = self.__qi[i]
                    self.__qi[i] += self.__lr * (err * self.__pu[u] - self.__lamda * self.__qi[i])
                    self.__pu[u] += self.__lr * (err * temp - self.__lamda * self.__pu[u])
            self.__lr *= 0.9

    def __rating(self,u,i):
        pre_rating = self.mean + self.__bu[u] + self.__bi[i] + np.dot(self.__qi[i],self.__pu[u])
        return pre_rating

    def predict(self,test_x):
        return np.array([self.__rating(u,i) for u,i in test_x])

    def report(self,predict_y,test_y):
        length = len(test_y)
        mae = np.sum(abs(test_y - predict_y)) / length
        rmse = np.sqrt(np.sum(np.power(test_y - predict_y,2)) / length)
        print("mae=%f,rmse=%f" % (mae, rmse))


        
if __name__ == '__main__':
    from recommend.data import datasets
    df = datasets.load_100k('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_split(df,10,10,0.2)
    svd = SVD(0.01,0.01,50,50)
    svd.fit(train_x,train_y)
    svd.report(svd.predict(test_x),test_y)
        
        
        
        
        
        
        
        
          