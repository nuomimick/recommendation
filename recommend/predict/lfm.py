from numpy.random import random
import numpy as np

class LFM:
    def __init__(self,lr,lamda,f,steps):
        self.__lr = lr
        self.__lamda = lamda
        self.__steps = steps
        self.__f = f
        self.__user_item = {}
        self.__item_user = {}
        self.p = {}
        self.q = {}

    def fit(self,train_x,train_y,test_x,test_y):
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

        for _ in range(self.__steps):
            lfm.report(lfm.predict(test_x),test_y)
            for u in self.__user_item:
                dict_items = self.__user_item[u]
                for i,r in dict_items.items():
                    e = r - np.dot(self.p[u],self.q[i])
                    tmp = self.q[i]
                    self.q[i] += self.__lr * (e * self.p[u] - self.__lamda * tmp)
                    self.p[u] += self.__lr * (e * tmp - self.__lamda * self.p[u])
            self.__lf *= 0.9
        print('iteration finished')

    def predict(self,test_x):
        return np.array([np.dot(self.p[u], self.q[i]) for u,i in test_x])

    def report(self,predict_y,test_y):
        length = len(test_y)
        mae = np.sum(abs(test_y - predict_y)) / length
        rmse = np.sqrt(np.sum(np.power((test_y - predict_y),2)) / length)
        print(mae,rmse)

if __name__ == '__main__':
    from recommend.data import datasets

    df = datasets.load_100k('pd').alldata
    train_x,test_x,train_y,test_y = datasets.filter_deal(df,10,10,0.2)

    lfm = LFM(0.015, 0.02, 50, 50)
    lfm.fit(train_x,train_y,test_x,test_y)
    lfm.report(lfm.predict(test_x),test_y)     