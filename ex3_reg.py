import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ex3 import sigmoid
import scipy.optimize as opt

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad

path = os.getcwd() + '/data/ex2data2.txt'
data2 = pd.read_csv(path, header = None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

data2.drop('Test 1', axis = 1, inplace = True)
data2.drop('Test 2', axis = 1, inplace = True)

cols = data2.shape[1]
X2 = data2.iloc[:, 1: cols]
y2 = data2.iloc[:, 0:1]

#convert to numpy arrays and initialize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

learningRate = 1

result2 = opt.fmin_tnc(func = costReg, x0 = theta2,
                       fprime = gradientReg, args = (X2, y2, learningRate))

print(result2)
#plot data
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['Test 1'], positive['Test 2'], s = 50, c = 'b', marker ='o', label='Accepted')
# ax.scatter(negative['Test 1'], negative['Test 2'], s = 50, c = 'r', marker ='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()