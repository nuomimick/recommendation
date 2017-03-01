import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BaseLoad:
	def __init__(self,type='np'):
		self.type = type
		self.dir = os.path.dirname(__file__)
	@property
	def data(self):
		df = self.df.loc[:,['user_id','item_id','timestamp']]
		if self.type == 'np':
			return np.array(df)
		elif self.type == 'pd':
			return df
	@property
	def target(self):
		df = self.df.loc[:,'rating']
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

	def __init__(self,type):
		super().__init__(type)
		path = os.path.join(self.dir,r'data\movielens\ml-100k\u.data')
		self.df = pd.read_csv(path,sep='\t',names=['user_id','item_id','rating','timestamp'])

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
	def __init__(self,type):
		super().__init__(type)
		path = os.path.join(self.dir,r'data\movielens\ml-1m\ratings.dat')
		self.df = pd.read_csv(path,sep='\t',names=['user_id','item_id','rating','timestamp'])

class load_10m:
	def __init__(self,type):
		super().__init__(type)
		path = os.path.join(self.dir,r'data\movielens\ml-10m\ratings.dat')
		reader = pd.read_csv(path,sep='::',names=['user_id','item_id','rating','timestamp'],\
			engine='python',chunksize=100000)
		self.df = pd.concat([df for df in reader])

class load_20m:
	def __init__(self,type):
		super().__init__(type)
		path = os.path.join(self.dir,r'data\movielens\ml-20m\ratings.csv')
		reader = pd.read_csv(path,sep=',',names=['user_id','item_id','rating','timestamp'],\
			engine='python',chunksize=100000)
		self.df = pd.concat([df for df in reader])


def filter_deal(data,ft_u,ft_i,test_size):
    df = data.groupby('user_id').filter(lambda x:len(x) > ft_u)
    df = df.groupby('item_id').filter(lambda x:len(x) > ft_i)
    print('filter finish')
    data = np.array(df.loc[:,['user_id','item_id']])
    target = np.array(df.loc[:,'rating'])
    train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=test_size)
    print('split finish')
    return (train_x,test_x,train_y,test_y)

	


	