'''
Created on 2017年2月26日
@author: zrh
@description:基于物品协同过滤的排序算法
'''
'''
from __future__ import division
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from operator import itemgetter
import numpy as np
import pandas as pd
import pickle as pk
import random
import os
import time
import sys

def addToVec(vec,key,value):
    vec.setdefault(key,value)
    
def addToMat(mat,i,j,r):
    mat.setdefault(i,{})
    mat[i][j] = r
    
def getData():
    if os.path.exists("train.pk"):
        with open("train.pk","rb") as f:
            data = pk.load(f)
    else:    
        filename = r"E:\movielens\ml-100k\ml-100k\u.data"
        header = ['userid','itemid','rating','timastamp']
        df = pd.read_csv(filename, sep = "\t", names = header)
        data = np.array(df)
        data = np.random.permutation(data)
        with open("train.pk","wb") as f:
            pk.dump(data,f)
    return data
#         trainData,testData = cv.train_test_split(df,test_size=0.2)
#         with open("train.pk","wb") as f:
#             pk.dump(trainData,f)
#         with open("test.pk","wb") as f:
#             pk.dump(testData,f)

def splitData(data,M,k,seed):
    test = []
    train = []
    random.seed(seed)
    for line in data:
        u,i,r = line[0],line[1],line[2]
        if random.randint(0,M) == k:
            test.append([u,i,r])
        else:
            train.append([u,i,r])
    return train,test

def commonUsers(item_user,i,j):
    if i in item_user and j in item_user:
        cusers = set(item_user[i].keys()) & set(item_user[j].keys())
        return cusers
    else:return {}

def cosine(user_item,item_user,i,j,ave_users,ave_items):
    users = commonUsers(item_user, i, j)
    if len(users) == 0:return 0
    user_i = item_user[i]
    user_j = item_user[j]
    sum_i,sum_j = 0.0,0.0
    for key,value in user_i.items():
        sum_i += value**2
    for key,value in user_j.items():
        sum_j += value**2
    s = 0.0
    for u in users:
        s += item_user[i][u]*item_user[j][u]    
    return s / np.sqrt(sum_i * sum_j)

def pearson(user_item,item_user,i,j,ave_users,ave_items):
    cusers = commonUsers(item_user, i, j)
    if len(cusers) == 0:return 0
    up = 0.0
    down1,down2 = 0.0,0.0
    for u in cusers:
        up += ((user_item[u][i] - ave_items[i][0]) * (user_item[u][j] - ave_items[j][0]))
        down1 += (user_item[u][i] - ave_items[i][0])**2
        down2 += (user_item[u][j] - ave_items[j][0])**2
    down = np.sqrt(down1 * down2)
    result = up / down if down > 0 else 0
    return result

def adjcosine(user_item,item_user,i,j,ave_users,ave_items):
    users = commonUsers(item_user, i, j)
    if len(users) == 0:return 0
    up = 0.0
    down1,down2 = 0.0,0.0
    for u in users:
        up += ((user_item[u][i] - ave_users[u]) * (user_item[u][j] - ave_users[u]))
        down1 += (user_item[u][i] - ave_users[u])**2
        down2 += (user_item[u][j] - ave_users[u])**2
    down = np.sqrt(down1 * down2)
    result = up / down if down > 0 else 0
    return result

def similarity(user_item,item_user,ave_users,ave_items):
    sim_dct = {}
    length = len(item_user)
    for index_i in range(length):
        i = list(item_user.keys())[index_i]
        for index_j in range(index_i+1,length):
            j = list(item_user.keys())[index_j]
            sim = adjcosine(user_item, item_user, i, j, ave_users, ave_items)
            addToMat(sim_dct, i, j, sim)
            addToMat(sim_dct, j, i, sim)
    return sim_dct
        
    

def predict(user_item,item_user,u,i,k,miu,ave_users,ave_items,sim):
    if u in user_item and i in item_user:
        items = user_item[u]
        wList = sim[i]
        wList = [x for x in wList.items() if x[0] in items and x[1] != 0]
        wList = sorted(wList,key=itemgetter(1),reverse=True)[:k]
        #特殊情况
        if len(wList) == 0:#item_user[i]不存在或者极少，使用用户均值
            return ave_users[u]
        s,sw = 0.0,0.0
        for item_w in wList:
            item,w = item_w[0],item_w[1]
            if ave_items[item][1] == 0:#处理标准差为0的情况，使用用户均值
                return ave_users[u]
            s += w * ((user_item[u][item] - ave_items[item][0]) / ave_items[item][1])
            sw += np.abs(w)
        result = ave_items[i][0] + ave_items[i][1] * s / sw
#         if result > 5:result = 5
#         elif result < 1:result = 1
        return result
    else:
        return miu
        

def init(trainData):
    user_item = {}
    item_user = {}
    miu = 0.0
    for line in trainData:
        u,i,r = line[0],line[1],line[2]
        addToMat(user_item, u, i, r)
        addToMat(item_user, i, u, r)
        miu += r
    return user_item,item_user,miu / trainData.shape[0]
    
    
def predictAll(trainData,testData):
#     if sys.argv[1] == "pearson":
#         func = pearson
#     elif sys.argv[1] == "cosine":
#         func = cosine
#     elif sys.argv[1] == "adjcosine":
#         func = adjcosine
    print("begin",time.ctime())
    user_item, item_user, miu = init(trainData)
    ave_users = {}
    ave_items = {}
    for u in user_item:
        r = user_item[u]
        s = [x[1] for x in r.items()]
        ave_users.setdefault(u,np.mean(s))
    for i in item_user:
        r = item_user[i]
        s = [x[1] for x in r.items()]
        ave_items.setdefault(i,[np.mean(s),np.std(s)])
    sim = similarity(user_item, item_user, ave_users, ave_items)
    print("sim finished!",time.ctime())       
    
    for k in range(10,110,10):
        s = 0.0
        for line in testData: 
            u,i,r = line[0],line[1],line[2]
            rui = predict(user_item, item_user, u, i, k,miu,ave_users,ave_items,sim)
            s += (r - rui)**2
        print(np.sqrt(s / testData.shape[0]))

if __name__ == "__main__":
    data = getData()
    kf = cv.KFold(data.shape[0],n_folds=5)
    for train,test in kf:
        trainData,testData = data[train],data[test]
        predictAll(trainData, testData)
    
'''
from array import array
import numpy as np

class ItemCR:
    def __init__(self,k,method='cosine'):
        self.__user_item = {}
        self.__item_user = {}
        self.__item_scores = {}
        self.__user_scores = {}
        self.sim_dct = None
        self.mean = 0.
        if method == 'cosine':
            self.__sim_method = self.__cosine
            

    def fit(self,train_x,train_y):
        item_list = {}
        user_list = {}
        self.mean = np.mean(train_y)
        m,n = train_x.shape
        for i in range(m):
            uid, iid, rating = train_x[i][0],train_x[i][1],train_y[i]
            self.__user_item.setdefault(uid,{})
            self.__user_item[uid][iid] = rating

            self.__item_user.setdefault(iid,{})
            self.__user_item[iid][uid] = rating

            item_list.setdefault(iid,array('i')).append(rating)
            user_list.setdefault(uid,array('i')).append(rating)

        for i in item_list:
            np_array = np.frombuffer(item_list[i])
            self.__item_scores[i] = (np.mean(np_array),np.std(np_array))
        for u in user_list:
            np_array = np.frombuffer(user_list[u])
            self.__user_scores[u] = (np.mean(np_array),np.std(np_array))

        self.sim_dct = self.__similarity()



    def predict(self,test_x):
        for index,value in enumerate(test_x):
            uid,iid = value[0],value[1]

    def report(self,test_y,predict_y):
        pass

    def __rating(self,u,i):
        if i in self.__item_user:
            items = self.__user_item[u]
            ws = self.sim_dct[i]
            ws = [x for x in ws.items() if x[0] in items and x[1] != 0]
            ws = sorted(ws,key=itemgetter(1),reverse=True)[:k]
            #特殊情况
            if len(ws) == 0:#item_user[i]不存在或者极少，使用用户均值
                return self.__user_scores[u][0]
            s,sw = 0.,0.
            for item_w in ws:
                item,w = item_w[0],item_w[1]
                if self.__item_scores[item][1] == 0:#处理标准差为0的情况，使用用户均值
                    return self.__user_scores[u]
                s += w * ((self.__user_item[u][item] - self.__item_scores[item][0]) / self.__item_scores[item][1])
                sw += np.abs(w)
            result = self.__item_scores[i][0] + self.__item_scores[i][1] * s / sw
            if result > 5:result = 5
            elif result < 1:result = 1
            return result
        else:
            return self.mean

    def __similarity(self):
        items = self.__item_user.keys()
        sim_dct = {}
        for i in items:
            sim_dct.setdefault(i,{})
        for i in range(len(items)):
            id_i = items[i]
            for j in range(i+1,len(items)):
                id_j = items[j]
                sim = self.__sim_method(id_i, id_j)
                sim_dct[id_i][id_j] = sim
                sim_dct[id_j][id_i] = sim
        return sim_dct 


    def __cosine(self,i,j):
        #i,j表示项目id
        cm_users = self.__commonUsers(i,j)
        if not cm_users:return 0
        up = 0.
        left,rigth = 0.,0.
        for u in cm_users:
            up += ((self.__user_item[u][i] - self.__item_scores[i][0]) 
                * (self.__user_item[u][j] - self.__item_scores[j][0]))
            left += (self.__user_item[u][i] - self.__item_scores[i][0])**2
            rigth += (self.__user_item[u][j] - self.__item_scores[j][0])**2
        down = np.sqrt(left * rigth)
        result = up / down if down > 0 else 0
        return result

    def __commonUsers(self,i,j):
        if i in self.__itemuser and j in self.__itemuser:
            cm_users = set(self.__itemuser[i].keys()) & set(self.__itemuser[j].keys())
            return cm_users
        else:
            return {}

if __name__ == '__main__':
    pass