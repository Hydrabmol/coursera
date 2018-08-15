import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


path = os.getcwd() + '/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
data.describe()
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))  

# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols] 

# convert from data frames to numpy matrices
X = np.matrix(X.values)  
y = np.matrix(y.values)

model = linear_model.LinearRegression()
model.fit(X, y)

x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  