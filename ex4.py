import numpy as np
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / np.exp(-z)

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def gradient_no_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    #intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel()
    # for i in range(parameters):
    #     term = np.multiply(error, X[:, i])
    #
    #     if (i == 0):
    #         grad[i] = np.sum(term) / len(X)
    #     else:
    #         grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])
    #
    # return grad

""" Multiclass Classifier """
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    #k X (n + 1) array for the parameters of each of the k Classifiers
    all_theta = np.zeros((num_labels, params + 1))

    #insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values = np.ones(rows), axis = 1)

    #labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun = cost, x0 = theta,
                        args= (X, y_i, learning_rate),
                        method = 'TNC', jac=gradient_no_loop)
        all_theta[i - 1, :] = fmin.x
