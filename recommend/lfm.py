'''
Created on 2016年9月4日

@author: Administrator
@description:lfm算法topN的实现
'''
'''
from __future__ import division
from sklearn.cross_validation import KFold
from numpy.random import random
from operator import itemgetter
import numpy as np
import pandas as pd
import pickle as pk
import os
import time
import random as rd
import math

class LFM:
    def __init__(self,trainMat,testMat):
        self.u2i = dict()
        self.i2u = dict()
        self.te_u2i = dict()
        items_pool = []#候选物品的列表，物品出现的次数和流行度成正比
        for line in trainMat:
            u,i = line[0],line[1]
            self.addToMat(self.u2i, u, i, 1)
            self.addToMat(self.i2u, i, u, 1)
            items_pool.append(i)
        self.selectNegativeSample(items_pool,3)#负样本与正样本的比例
        for line in testMat:
            u,i = line[0],line[1]
            self.addToMat(self.te_u2i, u, i, 1)
        
            
    def selectNegativeSample(self,items_pool,n):
        for u in self.u2i:
            items = self.u2i[u].keys()
            length = len(items)
            count = 0
            for i in range(10 * len(items)):
                item = items_pool[rd.randint(0,len(items_pool)-1)]
                if item not in items:
                    self.addToMat(self.u2i, u, item, 0)
                    self.addToMat(self.i2u, item, u, 0)
                    count += 1
                if count >= (n * length):
                    break
            
    def trainModel(self,gama,lamda,steps,f):
        self.q = dict()
        self.p = dict()
        for u in self.u2i:
            self.p.setdefault(u,random(f) / np.sqrt(f))
        for i in self.i2u:
            self.q.setdefault(i,random(f) / np.sqrt(f))
        for step in range(steps):
            t1 = time.time()
            for u in self.u2i:
                dict_items = self.u2i[u]
                for i,r in dict_items.items():
                    eui = r - self.predict(u,i)
                    temp = self.q[i]
                    self.q[i] += gama * (eui * self.p[u] - lamda * temp)
                    self.p[u] += gama * (eui * temp - lamda * self.p[u])
            gama *= 0.9
            
            rmse = self.trainRmse()
            t2 = time.time()
            print("第%d次迭代,耗时%d,rmse=%f" % (step,t2-t1,rmse))
    
    def recommend(self,user,N):
        rank = dict()
        #items = self.u2i[user]#包括用户喜欢和不喜欢的
        items = [x[0] for x in filter(lambda p:p[1]==1,self.u2i[user].items())]
        allItems = self.i2u.keys()
        for it in set(allItems) - set(items):
            rank[it] = self.predict(user, it)
        topN_items = sorted(rank.items(),key=itemgetter(1),reverse=True)[:N]
        topN_list = [x[0] for x in topN_items]
        return topN_list
    
    def testModel(self,N):
        sum_r1,sum_p1,sum_cm = 0,0,0
        for u in self.te_u2i:
            items = self.te_u2i[u]
            topN_list = self.recommend(u,N)
            sum_cm += len(set(items) & set(topN_list))
            sum_p1 += N
            sum_r1 += len(items)
        precision = sum_cm / sum_p1 #准确率
        recall = sum_cm / sum_r1 #召回率
        print("准确率=%f 召回率=%f" % (precision,recall))
    
    def trainRmse(self):
        s,n = 0,0.
        for u in self.u2i:
            items = self.u2i[u]
            for i,r in items.items():
                eui = r - self.predict(u, i)
                s += eui**2
                n += 1
        return math.sqrt(s / n)
    
    def addToMat(self,mat,i,j,r):
        mat.setdefault(i,{})
        mat[i][j] = r
        
    def predict(self,u,i):
        rui = np.sum(self.q[i]*self.p[u])
        return rui

def getData():
    if os.path.exists("data.pk"):
        with open("data.pk","rb") as f:
            data = pk.load(f)
    else:    
        filename = r"E:\movielens\ml-100k\ml-100k\u.data"
        header = ['userid','itemid','rating','timastamp']
        df = pd.read_csv(filename, sep = "\t", names = header)
        data = np.array(df)
        data = np.random.permutation(data)
        with open("data.pk","wb") as f:
            pk.dump(data,f)
    print("data read")
    return data

if __name__ == '__main__':
    data = getData()
    kf = KFold(data.shape[0],5)
    for train,test in kf:
        trainData,testData = data[train],data[test]
        lfm = LFM(trainData,testData)
        lfm.trainModel(0.015, 0.02, 30, 50)
        lfm.testModel(30)
'''
class LFM:
    def __init__(self,gamma,lamda,steps):
        pass
    def fit(self,train_x,train_y):
        pass
    def predict(self):
        pass
    def report(self):
        pass      