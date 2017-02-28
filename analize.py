from recommend import datasets
import matplotlib.pyplot as plt

df = datasets.load_100k('pd').data
df.groupby('item_id')['user_id'].count().sort_values().plot(kind='bar')
plt.show()
