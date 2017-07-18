import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

'''sigmoid function '''
def sigmoid(z):
    return 1 / (1 +np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis = 1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis = 1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

''' backprop function extends the cost function, so the cost function is edited here
    previously cost funciton '''
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    #reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, hidden_size + 1)))

    #run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    J = 0
    delta1 = np.zeros(theta1.shape)     #(25, 401)
    delta2 = np.zeros(theta2.shape)     #(10, 26) X * theta.T <- theta will be transposed, so input goes behind

    #compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    #account for regularization
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2))
          + np.sum(np.power(theta2[:, 1:], 2)))

    # end of cost function logic

    # begin backpropagation
    for t in range(m):
        a1t = a1[t,:]   #(1, 401)
        z2t = z2[t,:]   #(1, 25)
        a2t = a2[t,:]   #(1, 26)
        ht = h[t,:]     #(1, 10)
        yt = y[t,:]     #(1, 10)

        d3t = ht -yt    #(1, 10)
        z2t = np.insert(z2t, 0, values = np.ones(1)) #(1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t)) #(1, 26)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    #add gradient regularization term
    delta1[:,1:] = delta1[:, 1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:, 1:] + (theta2[:,1:] * learning_rate) / m

    #unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

''' computes gradient of the sigmoid function '''
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

#begin
data = loadmat('data/ex3data1.mat')

X = data['X']
y = data['y']

encoder = OneHotEncoder(sparse = false)
y_onehot = encoder.fit_transform(y)
print (y_onehot.shape)

#see what OneHotEncoder does
print(y[0], y_onehot[0, :])

'''intermediate step: see whether cost function is working '''

# initial setup
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# randomly initialize a parameter array of the size of the full network's  parameters
params = (np.random.random(size = hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

#unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels * (hidden_size + 1))))

print(theta1.shape, theta2.shape)

#run forward_propagate
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
print (a1.shape, z2.shape, a2.shape, z3.shape, h.shape)

#calculate cost function
print(
cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
)

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

print (J, grad.shape)

#minimize objective function (backpropagation) through scipy.optimize
fmin = minimize(fun = backprop, x0 = params, args = (input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
        method = 'TNC', jac = True, options = {'maxiter': 250})

print(fmin)

X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis = 1) + 1)
print (y_pred)

#compute accuracy
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = sum(map(int, correct)) / float(len(correct))
print ('accuracy = {0}%'.format(accuracy * 100))
