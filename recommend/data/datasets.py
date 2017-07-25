import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseLoad:
    def __init__(self, type='np'):
        self.type = type
        self.dir = os.path.dirname(__file__)

    @property
    def data(self):
        df = self.df.loc[:, ['user_id', 'item_id', 'timestamp']]
        if self.type == 'np':
            return np.array(df)
        elif self.type == 'pd':
            return df

    @property
    def target(self):
        df = self.df.loc[:, 'rating']
        if self.type == 'np':
            return np.array(df)
        elif self.type == 'pd':
            return df

    @property
    def alldata(self):
        df = self.df
        if self.type == 'np':
            return np.array(df)
        elif self.type == 'pd':
            return df


class load_100k(BaseLoad):
    def __init__(self, type):
        super().__init__(type)
        path = os.path.join(self.dir, r'movielens\ml-100k\u.data')
        self.df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # @property
    # def data(self):
    # 	# 第一种方法
    # 	# return np.loadtxt(self.path,dtype={'names':['user_id','item_id','timestamp'],
    # 	# 	'formats':['i4','i4','<i8']},usecols=[0,1,3])
    # 	return np.loadtxt(self.path,dtype=int,usecols=[0,1,3])

    # @property
    # def target(self):
    # 	return np.loadtxt(self.path,dtype=int,usecols=(2,))


class load_1m(BaseLoad):
    def __init__(self, type):
        super().__init__(type)
        path = os.path.join(self.dir, r'movielens\ml-1m\ratings.dat')
        self.df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])


class load_10m:
    def __init__(self, type):
        super().__init__(type)
        path = os.path.join(self.dir, r'movielens\ml-10m\ratings.dat')
        reader = pd.read_csv(path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], \
                             engine='python', chunksize=100000)
        self.df = pd.concat([df for df in reader])


class load_20m:
    def __init__(self, type):
        super().__init__(type)
        path = os.path.join(self.dir, r'movielens\ml-20m\ratings.csv')
        reader = pd.read_csv(path, sep=',', names=['user_id', 'item_id', 'rating', 'timestamp'], \
                             engine='python', chunksize=100000)
        self.df = pd.concat([df for df in reader])


def filter_split(data, ft_u, ft_i, test_size):
    # 过滤评分少于ft_u数量的用户
    df = data.groupby('user_id').filter(lambda x: len(x) > ft_u)
    # 过滤评分少于ft_i数量的物品
    df = df.groupby('item_id').filter(lambda x: len(x) > ft_i)
    data = np.array(df.loc[:, ['user_id', 'item_id']])
    target = np.array(df.loc[:, 'rating'])
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=test_size, random_state=0)
    return train_x, test_x, train_y, test_y


def load_musical_instruments():
    from collections import Counter
    ratingMatrixUI, ratingMatrixIU, reviewMatrix = {}, {}, {}
    first = False
    filepath = os.path.dirname(__file__) + '/musical_instruments/musical_instruments.arff'
    with open(filepath, 'r') as f:
        for line in f:
            if first:
                wCounter = Counter()
                u, i, r, review = line.strip().split(',')
                review = review.split(':')
                for rv in review:
                    wCounter[rv] += 1
                ratingMatrixUI.setdefault(u, {})
                ratingMatrixUI[u][i] = float(r)
                reviewMatrix.setdefault((u, i), wCounter)

                ratingMatrixIU.setdefault(i, {})
                ratingMatrixIU[i][u] = float(r)
                continue
            if '@DATA' in line:
                first = True
        return ratingMatrixUI, ratingMatrixIU, reviewMatrix


def load_musical_instruments_full():
    from collections import Counter
    ratingMatrixUI, ratingMatrixIU, reviewMatrix = {}, {}, {}
    first = False
    filepath = os.path.dirname(__file__) + '/musical_instruments/musical_instruments_full.arff'
    with open(filepath, 'r') as f:
        for line in f:
            if first:
                wCounter = Counter()
                u, i, r, review = line.strip().split(',')
                review = review.split(':')
                for rv in review:
                    wCounter[rv] += 1
                ratingMatrixUI.setdefault(u, {})
                ratingMatrixUI[u][i] = float(r)
                reviewMatrix.setdefault((u, i), wCounter)

                ratingMatrixIU.setdefault(i, {})
                ratingMatrixIU[i][u] = float(r)
                continue
            if '@DATA' in line:
                first = True
        return ratingMatrixUI, ratingMatrixIU, reviewMatrix


def split_musical_instruments():
    from collections import Counter
    def process(df):
        ratingMatrixUI, ratingMatrixIU, reviewMatrix = {}, {}, {}
        for idx, *line in df.itertuples():
            u, i, r, review = line
            wCounter = Counter()
            review = review.split(':')
            for rv in review:
                wCounter[rv] += 1
            ratingMatrixUI.setdefault(u, {})
            ratingMatrixUI[u][i] = float(r)
            reviewMatrix.setdefault((u, i), wCounter)

            ratingMatrixIU.setdefault(i, {})
            ratingMatrixIU[i][u] = float(r)
        return ratingMatrixUI, ratingMatrixIU, reviewMatrix

    first = False
    filepath = os.path.dirname(__file__) + '/musical_instruments/musical_instruments.arff'
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if first:
                line = line.strip().split(',')
                data.append(line)
                continue
            if '@DATA' in line:
                first = True
    df_data = pd.DataFrame(data, columns=['uid', 'iid', 'rating', 'review'])
    df_train, df_test = train_test_split(df_data,test_size=0.2,random_state=0)
    uidset = set(df_test.uid) - set(df_train.uid)
    if uidset:
        df_test = df_test[~df_test["uid"].isin(uidset)]
    iidset = set(df_test.iid) - set(df_train.iid)
    if iidset:
        df_test = df_test[~df_test["iid"].isin(iidset)]

    trainData = process(df_train)
    testData = process(df_test)
    return trainData, testData

if __name__ == '__main__':
    train,test = split_musical_instruments()
    print()
