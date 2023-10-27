import pandas as pd
import numpy as np
import cvxopt

C = [.001, .01, .1, 1, 10, 100, 1000]

spam_train_data = np.array(pd.read_csv('spam_train.data', header=None))

trainX = spam_train_data[:, :-1]
trainY = spam_train_data[:, -1]
trainY[trainY == 0] = -1

spam_validation_data = np.array(pd.read_csv('spam_validation.data', header=None))
validationX = spam_validation_data[:, :-1]
validationY = spam_validation_data[:, -1]
validationY[validationY == 0] = -1

spam_test_data = np.array(pd.read_csv('spam_test.data', header=None))
testX = spam_test_data[:, :-1]
testY = spam_test_data[:, -1]
testY[testY == 0] = -1

row, col = spam_train_data.shape


def calculate_accuracy(x, y, w, b):
    rows = len(x)
    count = 0
    for i in range(rows):
        pred = np.dot(w[:len(x[i])], x[i]) + b
        if pred * y[i] > 0:
            count += 1

    pred_accuracy = (count / rows) * 100
    return pred_accuracy


P = np.eye(row + col + 1)
P[:col, :col] = 1.0
P = cvxopt.matrix(P)

h = np.vstack((-np.ones((row, 1)), np.zeros((row, 1))))
h = cvxopt.matrix(h)

G = np.zeros((2 * row, row + col + 1))
for i in range(row):
    for j in range(trainX.shape[1]):
        G[i][j] = -1 * trainY[i] * trainX[i][j]

    G[i][col + i] = -1
    G[i][row + col] = -1 * trainY[i]
    G[row + i][col + i] = -1

G = cvxopt.matrix(G)


for c in C:
    Q = np.zeros((row + col + 1, 1))
    Q[col:row + col, 0] = c
    q = cvxopt.matrix(Q)

    result = cvxopt.solvers.qp(P, q, G, h)

    optimal_sol = result['x']

    w = []
    for i in range(len(optimal_sol)):
        w.append(optimal_sol[i])

    b = optimal_sol[row + col]

    accuracy = calculate_accuracy(trainX, trainY, w, b)
    print('Train accuracy: ', accuracy, ' for C =', c)
    accuracy = calculate_accuracy(validationX, validationY, w, b)
    print('Validation accuracy: ', accuracy, ' for value of C =', c)
    accuracy = calculate_accuracy(testX, testY, w, b)
    print('Test accuracy: ', accuracy, ' for value of C =', c)
