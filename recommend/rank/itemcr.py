from array import array
import numpy as np
import pickle as pk
from operator import itemgetter
import os
import time
from random import sample

class ItemCR:
    def __init__(self,k,sim_method='cosine',std_method='origin'):
        self.__k = k
        self.__user_item = {}
        self.__item_user = {}
        self.__item_scores = {}
        self.__user_scores = {}
        self.sim_dct = None
        self.__mean = 0.
        if sim_method == 'cosine':
            self.__sim_method = self.__cosine
        elif sim_method == 'adjcosine':
            self.__sim_method = self.__adjcosine
        elif sim_method == 'pearson':
            self.__sim_method = self.__pearson    

        if std_method == 'origin':
            self.__rating = self.__rating_origin
        elif std_method == 'center':
            self.__rating = self.__rating_center
        elif std_method == 'zscore':
            self.__rating = self.__rating_zscore
            

    def fit(self,train_x,train_y):
        item_dct = {}
        user_dct = {}
        self.__mean = np.mean(train_y)
        m,n = train_x.shape
        for i in range(m):
            uid, iid, rating = train_x[i][0],train_x[i][1],train_y[i]
            self.__user_item.setdefault(uid,{})
            self.__user_item[uid][iid] = rating

            self.__item_user.setdefault(iid,{})
            self.__item_user[iid][uid] = rating

            item_dct.setdefault(iid,array('i')).append(rating)
            user_dct.setdefault(uid,array('i')).append(rating)

        for i in item_dct:
            np_array = np.frombuffer(item_dct[i],dtype=np.int)
            self.__item_scores[i] = (np.mean(np_array),np.std(np_array))
        for u in user_dct:
            np_array = np.frombuffer(user_dct[u],dtype=np.int)
            self.__user_scores[u] = (np.mean(np_array),np.std(np_array))

        print('begin calculate similarity')
        filename = 'sim.pk'
        if os.path.exists(filename):
            with open(filename,'rb') as f:
                self.sim_dct = pk.load(f)
        else:
            t = time.time()
            self.sim_dct = self.__similarity()
            print("cost time %f" % (time.time() - t))
            with open(filename,'wb') as f:
                pk.dump(self.sim_dct,f)
        print('finish calculate similarity')

    def topN(self,test_x,top_n):
        all_items = set(self.__item_user)
        recommend_dict = {}
        for uid,iid in test_x:
            #unrating_items = sample(all_items - set(self.__user_item[uid]),30)
            unrating_items = sample(all_items - set(self.__user_item[uid]),30)
            ratings = [(iid,self.__rating(uid,iid)) for iid in unrating_items]
            ratings = sorted(ratings,key=itemgetter(1),reverse=True)[:top_n]
            recommend_dict.setdefault(uid,[tup[0] for tup in ratings])
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


    def __rating_origin(self,u,i):
        if u in self.__user_item and i in self.__item_user:
            items = self.__user_item[u]
            wt = self.sim_dct[i]
            wt = [x for x in wt.items() if x[0] in items]
            wt = sorted(wt,key=itemgetter(1),reverse=True)[:self.__k]
            s,abs_w = 0.,0.
            for j,w in wt:
                s += w * items[j]
                abs_w += abs(w)
            if abs_w == 0:return self.__mean
            result = s / abs_w
            if result > 5:result = 5
            elif result < 1:result = 1
        else:
            result = self.__mean
        return result

    def __rating_center(self,u,i):
        if u in self.__user_item and i in self.__item_user:
            items = self.__user_item[u]
            wt = self.sim_dct[i]
            wt = [x for x in wt.items() if x[0] in items]
            wt = sorted(wt,key=itemgetter(1),reverse=True)[:self.__k]
            s,abs_w = 0.,0.
            for j,w in wt:
                s += w * (items[j] - self.__item_scores[j][0])
                abs_w += abs(w)
            if abs_w == 0:return self.__mean
            result = s / abs_w + self.__item_scores[i][0] 
            if result > 5:result = 5
            elif result < 1:result = 1
        else:
            result = self.__mean
        return result

    def __rating_zscore(self,u,i):
        if u in self.__user_item and i in self.__item_user:
            items = self.__user_item[u]
            wt = self.sim_dct[i]
            wt = [x for x in wt.items() if x[0] in items]
            wt = sorted(wt,key=itemgetter(1),reverse=True)[:self.__k]
            s,abs_w = 0.,0.
            for j,w in wt:
                s += w * (items[j] - self.__item_scores[j][0]) / self.__item_scores[j][1]
                abs_w += abs(w)
            if abs_w == 0:return self.__mean
            result = s / abs_w * self.__item_scores[i][1] + self.__item_scores[i][0]
            if result > 5:result = 5
            elif result < 1:result = 1
        else:
            result = self.__mean
        return result

    def __similarity(self):
        items = list(self.__item_user.keys())
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
        users = self.__commonUsers(i, j)
        if len(users) == 0:return 0
        users_i = self.__item_user[i]
        users_j = self.__item_user[j]

        sum_lr = np.sum(np.power(np.array(list(users_i.values())),2)) * \
                np.sum(np.power(np.array(list(users_j.values())),2))
        
        sum_u = 0.
        for u in users:
            sum_u += users_i[u] * users_j[u]    
        sum_lr = sum_u / np.sqrt(sum_lr)
        return sum_u / sum_lr

    def __adjcosine(self,i,j):
        users = self.__commonUsers(i, j)
        if len(users) == 0:return 0
        users_i = self.__item_user[i]
        users_j = self.__item_user[j]

        sum_up = 0.
        sum_l,sum_r = 0.,0.
        for u in users:
            sum_up += (users_i[u] - self.__user_scores[u][0]) * (users_j[u] - self.__user_scores[u][0])
            sum_l += (users_i[u] - self.__user_scores[u][0])**2
            sum_r += (users_j[u] - self.__user_scores[u][0])**2
        sum_lr = np.sqrt(sum_l * sum_r)
        result = sum_up / sum_lr if sum_lr > 0 else 0#common users is little
        return result

    def __pearson(self,i,j):
        cusers = self.__commonUsers(i, j)
        if len(cusers) == 0:return 0
        users_i = self.__item_user[i]
        users_j = self.__item_user[j]

        sum_up = 0.
        sum_l,sum_r = 0.,0.
        for u in cusers:
            sum_up += ((users_i[u] - self.__item_scores[i][0]) * (users_j[u] - self.__item_scores[j][0]))
            sum_l += (users_i[u] - self.__item_scores[i][0])**2
            sum_r += (users_j[u] - self.__item_scores[j][0])**2
        sum_lr = np.sqrt(sum_l * sum_r)
        result = sum_up / sum_lr if sum_lr > 0 else 0
        return result

    def __commonUsers(self,i,j):
        cm_users = set(self.__item_user[i].keys()) & set(self.__item_user[j].keys())
        return cm_users

if __name__ == '__main__':
    from recommend.data import datasets

    df = datasets.load_100k('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_deal(df,20,20,0.2)

    ir = ItemCR(10,'cosine','origin')
    ir.fit(train_x,train_y)
    ir.report(test_x,10)



