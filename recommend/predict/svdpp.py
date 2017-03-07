'''
Created on 2016年9月4日

@author: Administrator
@description:svdpp算法的实现
'''
from __future__ import division
from sklearn import cross_validation as cv
from numpy.random import random
import numpy as np
import pandas as pd
import pickle as pk
import os
import time
def getData():
    if os.path.exists("train.pk") and os.path.exists("test.pk"):
        with open("train.pk","rb") as f:
            trainData = pk.load(f)
        with open("test.pk","rb") as f:
            testData = pk.load(f)
    else:    
        filename = "E:\\movielens\\ml-100k\\ml-100k\\u.data"
        header = ['userid','itemid','rating','timastamp']
        df = pd.read_csv(filename, sep = "\t", names = header)
        trainData,testData = cv.train_test_split(df,test_size=0.2)
        with open("train.pk","wb") as f:
            pk.dump(trainData,f)
        with open("test.pk","wb") as f:
            pk.dump(testData,f)
    return trainData,testData

def addToMat(mat,i,j,r):
    mat.setdefault(i,{})
    mat[i][j] = r

def init(trainData):
    user_item = dict()
    item_user = dict()
    miu = 0.0
    for line in trainData.itertuples():
        u,i,r = line[1],line[2],line[3]
        addToMat(user_item, u, i, r)
        addToMat(item_user, i, u, r)
        miu += r
    return user_item,item_user,miu / trainData.shape[0]


def predict(user_item,item_user, u, i, miu, bu, bi,q,s):
    #print(miu,bu[u],bi[i],np.sum(q[i]*(p[u] + s)))
    if u in user_item and i in item_user:
        rui = miu + bu[u] + bi[i] + np.sum(q[i]*s)
    else:rui = miu
    return rui

def sumy(user_item,u,p,y):
    dict_items = user_item[u]
    s = 0
    for j in dict_items.keys():
        s += y[j]
    return s / np.sqrt(len(dict_items)) + p[u]

def train(user_item,item_user,miu):
    bu = dict()
    bi = dict()
    q = dict()
    p = dict()
    y = dict()
    f = 200
    for u in user_item.keys():
        bu.setdefault(u,0)
        p.setdefault(u,random(f) / np.sqrt(f))
        for i in user_item[u].keys():
            bi.setdefault(i,0)
            q.setdefault(i,random(f) / np.sqrt(f))
            y.setdefault(i,random(f) / np.sqrt(f))
            
    gama = 0.009
    lamda = 0.005
    lamda1 = 0.015
    k = 30
    for t in range(k):
        for u in user_item:
            dict_items = user_item[u]
            sums = 0
            s = sumy(user_item, u, p, y)
            for i,r in dict_items.items():
                eui = r - predict(user_item,item_user,u,i,miu,bu,bi,q,s)
                bu[u] += gama * (eui - lamda * bu[u])
                bi[i] += gama * (eui - lamda * bi[i])
                temp = q[i]
                sums += temp * eui
                q[i] += gama * (eui * s - lamda1 * temp)
                p[u] += gama * (eui * temp - lamda1 * p[u])
            for j in dict_items.keys():
                y[j] += gama * (sums / np.sqrt(len(dict_items)) - lamda1 * y[j])
        gama *= 0.9
    return bu,bi,q,p,y

def test():
    trainData, testData = getData()
    user_item,item_user,miu= init(trainData)
    bu,bi,q,p,y = train(user_item,item_user,miu)
    print("训练完成！")
    test_user2item,test_item2user,test_miu = init(testData)
    s = 0.0
    for u in test_user2item:
        ss = sumy(user_item, u, p, y)
        for i,r in test_user2item[u].items():          
            eui = r - predict(user_item,item_user, u, i, miu, bu, bi,q,ss)
            s += eui ** 2
    rmse = np.sqrt(s / testData.shape[0])
    print("rmse: %f" % rmse)

if __name__ == '__main__':
    for i in range(5):
        test()