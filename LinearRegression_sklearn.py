import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = os.getcwd() + '/data/ex1data1.txt'
data = pd.read_csv(path, header=None, names =['Population', 'Profit'])
print (data.describe())
# data.plot(kind= 'scatter', x = 'Population', y = 'Profit', figsize = (12,8))

#append a ones coliumn to the front of data
data.insert(0, 'Ones', 1)

#set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0 : cols - 1]

y = data.iloc[:, cols - 1 : cols]

#finally convert data frames to numpy matrices  and instantiate parameter matrix
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0,0])

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(x, f, 'r', label = "Prediction")
ax.scatter(data.Population, data.Profit, label = "Training Data")
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population Size')
plt.show()