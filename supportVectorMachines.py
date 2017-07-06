import numpy as np
import pandas as import pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

raw_data = loadmat('data/ex6data1.mat')
print(raw_data)

data = pd.DataFrame(raw_data['X'], columns = ['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], s = 50, marker = 'o', label = 'Positive')
ax.scatter(negative['X1'], negative['X2'], s = 50, marker = 'x', label = 'Negative')
ax.legend()

# Train SVM classifier here
svc = svm.LinearSVC(C = 1, loss = 'hinge', max_iter = 1000)
print (svc)
print (svc.fit(data[['X1', 'X2']], data['y']))
print (svc.score(data[['X1', 'X2']], data['y']))

#Try with a larger value of C (1 / lambda)
svc2 = svm.LinearSVC(C = 100, loss = 'hinge', max_iter = 1000)
svc2.fit(data[['X1', 'X2']], data['y'])
svc2.score(data[['X1', 'X2']], data['y'])

#plot svc1 data confidence (C = 1)
data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s = 50, c = data['SVM 1 Confidence'], cmap = 'seismic')
ax.set_title('SVM (C = 1) Decision Confidence')

#plot svc2 data confidence (C = 100)
data['SVM 2 Confidence'] = svc.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(data['X1'], data['X2'], s = 50, c = data['SVM 2 Confidence'], cmap = 'seismic')
ax.set_title('SVM (C = 100) Decision Confidence)

x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2
gaussian_kernel(x1, x2, sigma)


#data set with non-linear decision boundary
raw_data = loadmat('data/ex6data2.mat')
data = pd.DataFrame(raw_data['X'], columns = ['X1', 'X2'])
data['y'] = raw_data['y']
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize = (12, 8))
ax.scatter(positive['X1'], positive['X2'], s = 30, marker = 'o', label = 'Positive')
ax.scatter(negative['X1'], negative['X2'], s = 30, marker = 'x', label = 'Negative')
ax.legend()

svc = svm.SVC(C = 100, gamma = 10, probability = True)
svc.fit(data[['X1', 'X2']], data['y'])
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:, 0]

fig, ax = plt.subplots(figsize = (12,8))
ax.scatter(data['X1'], dat['X2'], s = 30, c = data['Probability'], cmap = 'Reds')
