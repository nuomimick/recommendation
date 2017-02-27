from recommend import loaddata

data = loaddata.load_1m()
print(data.data.shape)
print(data.target.shape)