import os
import numpy as np
import pandas as pd

class BaseLoad:
	def __init__(self):
		self.df = None
	@property
	def data(self):
		return np.array(self.df.loc[:,['user_id','item_id','timestamp']])
	@property
	def target(self):
		return np.array(self.df.loc[:,'rating'])

class load_100k():

	def __init__(self):
		self.path = os.path.join(os.getcwd(),r'recommend\data\movielens\ml-100k\u.data')

	@property
	def data(self):
		# 第一种方法
		# return np.loadtxt(self.path,dtype={'names':['user_id','item_id','timestamp'],
		# 	'formats':['i4','i4','<i8']},usecols=[0,1,3])
		return np.loadtxt(self.path,dtype=int,usecols=[0,1,3])

	@property
	def target(self):
		return np.loadtxt(self.path,dtype=int,usecols=(2,))

class load_1m(BaseLoad):
	def __init__(self):
		path = os.path.join(os.getcwd(),r'recommend\data\movielens\ml-1m\ratings.dat')
		self.df = pd.read_csv(path,sep='\t',names=['user_id','item_id','rating','timestamp'])

class load_10m:
	def __init__(self):
		path = os.path.join(os.getcwd(),r'recommend\data\movielens\ml-10m\ratings.dat')
		reader = pd.read_csv(path,sep='::',names=['user_id','item_id','rating','timestamp'],\
			engine='python',chunksize=100000)
		self.df = pd.concat([df for df in reader])

class load_20m:
	def __init__(self):
		path = os.path.join(os.getcwd(),r'recommend\data\movielens\ml-20m\ratings.csv')
		reader = pd.read_csv(path,sep=',',names=['user_id','item_id','rating','timestamp'],\
			engine='python',chunksize=100000)
		self.df = pd.concat([df for df in reader])


	


	