from __future__ import print_function 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import library as lb
import sklearn.tree as tr
import matplotlib.pyplot as pl
from PIL import Image



df = pd.read_csv('Housing_data_price.csv', index_col = 0, parse_dates = True)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size=0.7, shuffle = False)    


print(xtest)
tree = lb.DecisionTreeID3(max_depth = 4, min_samples_split = 2)
tree.fit(xtrain,ytrain)
data = tree.predict(xtest)
y_test = np.array(ytest)
dem = 0
for i in range(len(data)):
	if(data[i] == y_test[i]):
		dem = dem + 1
	print('Dự đoán [', data[i], '] Thực tế là [', y_test[i], ']\n')
print("Thuật toán ID3 dự đoán đúng được ",dem,' giá trị trên tổng số ',len(data))

# for i, j in enumerate(X):
# 	for k in range(3,10):
# 		if(j[k] == "yes"):
# 			j[k] = 3
# 		elif(j[k] == "no"):
# 			j[k] = 1
# 		elif(j[k] == "furnished"):
# 			j[k] = 4
# 		elif(j[k] == "semi-furnished"):
# 			j[k] = 5
# 		elif(j[k] == "unfurnished"):
# 			j[k] = 6

# dtree = tr.DecisionTreeClassifier()
# dtree = dtree.fit(X,y)
# fig =pl.figure(figsize =(20,25))
# _=tr.plot_tree(dtree)
# # fig.savefig('cay.png') #lưu vào tập tin cay.png
# image = Image.open('cay.png')
# image.show()

