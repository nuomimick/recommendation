from recommend.data import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = datasets.load_100k('pd').alldata
df = df.groupby('user_id').filter(lambda x:len(x) > 20)
df = df.groupby('item_id').filter(lambda x:len(x) > 20)
print('finish')
data = df.loc[:,['user_id','item_id']]
target = df.loc[:,'rating']
train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.2,stratify=target)
print(train_x.item_id.unique().shape)
print(test_x.item_id.unique().shape)
print(len(train_x[train_x['item_id'] == 739]))
test_x.groupby('item_id')['user_id'].count().sort_values().plot(kind='bar')
plt.show()
print('finish')



