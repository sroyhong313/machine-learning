def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd() + '/data/ex1data1.txt'
data = pd.read_csv(path, header=None, names =['Population', 'Profit'])
print (data.describe())
data.plot(kind= 'scatter', x = 'Population', y = 'Profit', figsize = (12,8))
#plt.show()

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

print (computeCost(X,y,theta))

#initialize learning rate alpha and number of iterations
alpha = 0.01
iters = 1000

#perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + (g[0,1] * x)

fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(x, f, 'r', label = 'Prediction')
ax.scatter(data.Population, data.Profit, label = 'Training Data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population Size')
plt.show()