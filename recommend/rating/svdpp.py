from __future__ import division
from numpy.random import random
import numpy as np
import math
import pandas as pd
import pickle as pk
import os
import time

class SVDPP:

    def __init__(self,lr,lamda1,lamda2,ft,steps):
        '''
        lr: learing rate
        lamda: egularization factor
        ft:factor dimension
        '''
        self.__lr = lr
        self.__lamda1 = lamda1
        self.__lamda2 = lamda2
        self.__ft = ft
        self.__steps = steps
        self.__user_item = {}
        self.__item_user = {}
        self.__pu = {}
        self.__qi = {}
        self.__y = {}
        self.__bu = {}
        self.__bi = {}
        self.mean = 0.

    def fit(self,train_x,train_y):
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
            self.__y.setdefault(i,random(self.__ft) / math.sqrt(self.__ft))
            
        for _ in range(self.__steps):
            self.report(self.predict(train_x),train_y)
            for u in self.__user_item:
                items = self.__user_item[u]
                sums = 0.
                s = self.__y_sum(u)
                for i,r in items.items():
                    err = r - self.__rating(u,i,s)
                    self.__bu[u] += self.__lr * (err - self.__lamda1 * self.__bu[u])
                    self.__bi[i] += self.__lr * (err - self.__lamda1 * self.__bi[i])
                    temp = self.__qi[i]
                    sums += temp * err
                    self.__qi[i] += self.__lr * (err * s - self.__lamda2 * temp)
                    self.__pu[u] += self.__lr * (err * temp - self.__lamda2 * self.__pu[u])
                for j in items.keys():
                    self.__y[j] += self.__lr * (sums / np.sqrt(len(items)) -
                                                self.__lamda2 * self.__y[j])
            self.__lr *= 0.9

    def __rating(self, u, i, s):
        rating = self.mean + self.__bu[u] + self.__bi[i] + np.dot(self.__qi[i], s)
        return rating

    def __y_sum(self,u):
        r = np.sum([self.__y[j] for j in self.__user_item[u].keys()],axis=0)
        return r / math.sqrt(len(self.__user_item[u])) + self.__pu[u]

    def predict(self,test_x):
        predict_y = []
        for uid,iid in test_x:
            s = self.__y_sum(uid)
            predict_y.append(self.__rating(uid,iid,s))
        return predict_y

    def report(self,predict_y,test_y):
        length = len(test_y)
        mae = np.sum(abs(test_y - predict_y)) / length
        rmse = np.sqrt(np.sum(np.power(test_y - predict_y,2)) / length)
        print("mae=%f,rmse=%f" % (mae, rmse))

if __name__ == '__main__':
    from recommend.data import datasets
    df = datasets.load_100k('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_split(df,10,10,0.2)

    svd = SVDPP(0.009,0.005,0.015,50,50)
    svd.fit(train_x,train_y)
    svd.report(svd.predict(test_x),test_y)