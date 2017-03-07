'''
Created on 2016年10月31日

@author: Administrator
'''
from __future__ import division
import numpy as np
import pandas as pd
from copy import deepcopy
from random import sample
from math import exp, log, sqrt
from numpy.random import random
from numpy.linalg.linalg import norm
from LTR.rank_metrics import ndcg_at_k,err_at_k
import time

class RankList(object):
    '''
    classdocs
    '''
    tr_u2i = {}
    tr_i2u = {}
    te_u2i = {}
    U = {}
    V = {}
    def __init__(self, data, n):
        '''
        Constructor
        '''
        u2i = {}
        self.all_items = data.itemid.unique()#item数量
        self.all_users = data.userid.unique()#users数量
        self.N = n
        datMat = np.array(data)
        for line in datMat:
            u,i,r = line[0],line[1],line[2]
            u2i.setdefault(u,{})
            u2i[u][i] = r
        
        #生成训练集并去除训练集中的item
        cpdict1 = deepcopy(u2i)    
        for u in cpdict1:
            items = u2i[u]
            self.tr_u2i.setdefault(u,{})
            for i in sample(items.keys(),n):
                self.tr_u2i[u][i] = items[i]
                self.tr_i2u.setdefault(i,{})
                self.tr_i2u[i][u] = items[i]
                del(u2i[u][i])
                
        print("训练集生成完成!")
        #生成测试集        
        cpdict2 = deepcopy(u2i)    
        for u in cpdict2:
            items = u2i[u]
            self.te_u2i.setdefault(u,{})
            for i in sample(items.keys(),n):
                self.te_u2i[u][i] = items[i]
                del(u2i[u][i])
        print("测试集生成完成!")
                
        #生成候选集(往测试集里加不相关项)
        for u in self.te_u2i:
            for i in sample(set(self.all_items) - set(cpdict1[u]),1000):
                self.te_u2i[u][i] = 0 
        print("候选集生成完成！")
    
    #a代表步长，b代表正则化系数,c代表维数,d代表迭代次数    
    def train(self,a,b,c,d):
        for u in self.all_users:
            self.U.setdefault(u,random(c))
        for i in self.all_items:
            self.V.setdefault(i,random(c))
        
        for step in range(d):
            midRst = dict()
            for u in self.tr_u2i:
                s1,s2= 0.,0.
                for i,r in self.tr_u2i[u].items():
                    s1 += exp(self.gfunc(u, i))
                    s2 += exp(r)
                midRst[u] = [s1,s2]
                s3 = 0.    
                for i,r in self.tr_u2i[u].items():
                    s3 += (exp(self.gfunc(u, i)) / s1 - exp(r) / s2)*self.gdfunc(u, i)*self.V[i]
                self.U[u] -= a * (s3 + b * self.U[u])
                
            for i in self.tr_i2u:
                s3 = 0.    
                for u,r in self.tr_i2u[i].items():
                    s3 += (exp(self.gfunc(u, i)) / midRst[u][0] - exp(r) / midRst[u][1])*self.gdfunc(u, i)*self.U[u]
                self.V[i] -= a * (s3 + b * self.V[i])
            print("第%d次迭代完成！" % (step+1))
            self.evaluate(5)
            #误差
#             s4 = 0.
#             for u in self.tr_u2i:
#                 s3 = 0.
#                 for i,r in self.tr_u2i[u].items():
#                     s3 += -(exp(r) / midRst[u][1]) * log(exp(self.gfunc(u, i))/midRst[u][0])
#                 s4 += s3
#             m = []
#             for u in self.U:
#                 m.append(self.U[u])
#             n = []
#             for v in self.V:
#                 n.append(self.V[v])
#             print("%f" % (s4 + b * 0.5 * (norm(m,"fro")**2 + norm(n,"fro")**2)))
                           
    
    def gfunc(self,i,j):
        x = sum(self.U[i]*self.V[j])
        return 1 / (1 + exp(-x))
        
    def gdfunc(self,i,j):
        x = sum(self.U[i]*self.V[j])
        return exp(-x) / (1+exp(-x))**2            
    
    def evaluate(self,N):
        merr,ndcg=0.,0.
        for u in self.te_u2i:
            rList = [(i,sum(self.U[u]*self.V[i])) for i in self.te_u2i[u]]
            nrank = sorted(rList,key=lambda p:p[1],reverse=True)[:N]
            ratings = [self.te_u2i[u][x[0]] for x in nrank]
            merr += err_at_k(ratings, self.N)
            ndcg += ndcg_at_k(ratings, self.N)
        print(merr / len(self.te_u2i),ndcg / len(self.te_u2i))
    
        
        
        
filename = r"E:\data\movielens\ml-1m\ratings.dat"
header = ['userid','itemid','rating','timastamp']
df = pd.read_csv(filename, sep = "\t", names = header)
rl = RankList(df,5) 
rl.train(0.01, 0.01, 5, 30)    