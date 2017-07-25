import pandas as pd
from random import randint,random
import numpy as np
import lda
import lda.datasets

class ldaRc:
    def __init__(self,topics,iters):
        self.topics = topics
        self.iters = iters
        self.__user_item = {}
        self.__item_user = {}
        self.items_pool = []

    #negative sample
    def selectNegativeSample(self,n):
        for u in self.__user_item:
            items = self.__user_item[u].keys()
            length = len(items)
            count = 0
            for i in range(10 * length):
                item = self.items_pool[randint(0,len(self.items_pool)-1)]
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
        weight = pd.DataFrame(self.__item_user).fillna(0)#行是用户，列是电影
        self.rows = dict(zip(weight.index,range(weight.shape[0])))#用户与行对应关系
        self.columns = dict(zip(range(weight.shape[1]),weight.columns))#列与电影对应关系
        self.lda(np.array(weight,dtype=int),self.topics,self.iters)


    def lda(self,weight, topics, iters):
        # LDA算法
        model = lda.LDA(n_topics=topics, n_iter=iters, random_state=1)
        model.fit(np.asarray(weight))  # model.fit_transform(X) is also available
        self.topic_word = model.topic_word_  # model.components_ also works

        # 文档-主题（Document-Topic）分布
        self.doc_topic = model.doc_topic_


    def sample_interests(self,distributation,times):
        sample_list = []
        for _ in range(times):
            accum_p = 0.
            sp = random()
            for idx,p in enumerate(distributation):
                accum_p += p
                if sp < accum_p and idx not in sample_list:
                    sample_list.append(idx)
                    break
        return sample_list

    def sample_items(self,distributation,times,rating_items):
        sample_list = []
        for _ in range(times):
            accum_p = 0.
            sp = random()
            for idx, p in enumerate(distributation):
                accum_p += p
                if sp < accum_p and idx not in sample_list and self.columns[idx] not in rating_items:
                    sample_list.append(idx)
                    break
        return sample_list


    def topN(self,test_x,top_n):
        recommend_dict = {}
        df = pd.DataFrame(test_x,columns=['user','item'])
        for user,items in df.groupby('user'):
            #interests = self.sample_interests(self.doc_topic[self.rows[user]],2)#采样2个兴趣,不重复
            interests = [self.doc_topic[self.rows[user]].argmax()]#兴趣最大值
            recommend_items = []
            for i in interests:
                recommend_items.extend(self.sample_items(self.topic_word[i], top_n, self.__user_item[user]))
            recommend_dict.setdefault(user,recommend_items)
        return recommend_dict

    def evaluate(self,test_x,test_y):
        R = np.matrix(self.doc_topic) * np.matrix(self.topic_word)
        print(R)
        m = len(test_x)
        for i in range(m):
            u,i,r = test_x[i][0],test_x[i][1],test_y[i]

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

if __name__ == '__main__':
    from recommend.data import datasets

    df = datasets.load_100k('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_split(df,20,20,0.2)

    ld = ldaRc(20,100)
    ld.fit(train_x,train_y)
    ld.evaluate(test_x,test_y)
    # ld.report(test_x)