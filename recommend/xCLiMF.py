'''
Created on 2016年11月7日

@author: Administrator
'''
from __future__ import division 

from copy import deepcopy
from math import exp
from random import sample
import time

from numpy.random import random

from LTR.rank_metrics import err_at_k, ndcg_at_k, average_precision
import numpy as np
import pandas as pd


class xCLiMF(object):
    '''
    classdocs
    '''
    U = {}
    V = {}
    tr_u2i = {}#训练集key为user
    te_u2i = {}#测试集

    def __init__(self, data,n):
        '''
        Constructor
        data原始数据矩阵
        n表示每个用户的测试集数量
        '''
        u2i = {}
        pop_items = {}
        self.all_items = data.itemid.unique()#item数量
        self.all_users = data.userid.unique()#users数量
        datMat = np.array(data)
        for line in datMat:
            u,i,r = line[0],line[1],line[2]
            u2i.setdefault(u,{})
            u2i[u][i] = r
            pop_items.setdefault(i,0)
            pop_items[i] += 1
        
        self.top3Items = [x[0] for x in sorted(pop_items.items(),key=lambda p:p[1],reverse=True)[:3]]
        
        #生成测试集        
        cpdict1 = deepcopy(u2i)    
        for u in cpdict1:
            items = u2i[u]
            self.te_u2i.setdefault(u,{})
            for i in sample(items.keys(),5):
                self.te_u2i[u][i] = items[i]
                del(u2i[u][i])
        print("测试集生成完成!")
        
        #生成训练集并去除训练集中的item
        cpdict2 = deepcopy(u2i)    
        for u in cpdict2:
            items = u2i[u]
            self.tr_u2i.setdefault(u,{})
            for i in sample(items.keys(),n):
                self.tr_u2i[u][i] = items[i]
                del(u2i[u][i])
        print("训练集生成完成!")
                
        #生成候选集(往测试集里加不相关项)
        for u in self.te_u2i:
            for i in sample(set(self.all_items) - set(cpdict1[u]),100):
                self.te_u2i[u][i] = 0 
        print("候选集生成完成！")   
    
    def precompute_f(self,items_u,u):
        f = dict((j,sum(self.U[u]*self.V[j])) for j in items_u)
        return f

    def precompute_r(self,items_u):
        r = dict((j,(2**r - 1) / 32) for j,r in items_u.items())
        return r

    #a代表步长，b代表正则化系数,c代表维数,d代表迭代次数      
    def  train(self,a,b,c,d):
        #随机初始化
        for u in self.all_users:
            self.U.setdefault(u,0.01*random(c))
        for i in self.all_items:
            self.V.setdefault(i,0.01*random(c))
            
        for step in range(d):
            t1 = time.time()
            for u in self.tr_u2i:
                dU = 0.
                items = self.tr_u2i[u]
                f = self.precompute_f(items,u)
                r = self.precompute_r(items)
                for i in items:
                    for k in items:
                        dU += r[k]*self.dg(f[k] - f[i]) * (self.V[i]-self.V[k]) / (1 - r[k] * self.g(f[k] - f[i])) 
                    dU += r[i] * (self.g(-f[i]) * self.V[i] + dU)
                dU -= b * self.U[u]
                self.U[u] += a * dU
                
                for i in items:
                    dV = 0.
                    for k in items:
                        dV += r[k]*self.dg(f[i] - f[k]) * (1 / (1-r[k] * self.g(f[k]-f[i])) 
                                                                   - 1 / (1-r[i]*self.g(f[i]-f[k])))
                    dV = r[i]*(self.g(-f[i]) + dV)*self.U[u]-b*self.V[i]
                    self.V[i] += a * dV
                
            merr,ndcg,map = self.evaluate()
            t2 = time.time() - t1
            print("第%d次迭代，花费时间%ds，平均err=%f,ndcg=%f,map=%f" % (step+1,t2,merr,ndcg,map))
           
    def g(self,x):
        return 1 / (1 + exp(-x))
        
    def dg(self,x):
        return exp(-x) / (1 + exp(-x))**2 
    
    def evaluate(self):
        merr,ndcg,Map = 0., 0.,0.
        for u in self.te_u2i:
            items = self.te_u2i[u]
            f = self.precompute_f(items,u)
            rank = [(i,f[i]) for i in items if i not in self.top3Items]
            nrank = sorted(rank,key=lambda p:p[1],reverse=True)[:5]
            ratings = [items[x[0]] for x in nrank]
            merr += err_at_k(ratings, 5)
            ndcg += ndcg_at_k(ratings, 5)
            Map += average_precision(ratings,5)
        return merr / len(self.te_u2i),ndcg / len(self.te_u2i) ,Map / len(self.te_u2i)
                
        
filename = r"E:\data\movielens\ml-100k\u.data"
header = ['userid','itemid','rating','timastamp']
df = pd.read_csv(filename, sep = "\t", names = header)
xclimf = xCLiMF(df,10)
xclimf.train(0.001, 0.001, 10, 60)