from recommend import datasets
import matplotlib.pyplot as plt

df = datasets.load_1m('pd').data
df.groupby('user_id')['item_id'].count().sort_values().plot(kind='bar')
plt.show()
