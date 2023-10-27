import numpy as np
import pandas as pd
from cvxopt import matrix, solvers


def gaussian_kernel(x, z, sigma):
    return np.exp(-np.linalg.norm(x - z) ** 2 / (2 * sigma ** 2))


def train_svm(trainX, trainY, sigma, C):
    row, col = trainX.shape
    K = np.zeros((row, row))

    for i in range(row):
        for j in range(row):
            K[i, j] = gaussian_kernel(trainX[i], trainX[j], sigma)

    P = matrix(np.outer(trainY, trainY) * K)
    q = matrix(-np.ones((row, 1)))
    G = matrix(np.vstack((-np.eye(row), np.eye(row))))
    h = matrix(np.hstack((np.zeros(row), np.ones(row) * C)))
    A = matrix(trainY.reshape(1, -1))
    b = matrix(np.zeros(1))

    solution = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})

    a = np.ravel(solution['x'])

    sv_idx = a > 1e-4
    a = a[sv_idx]
    sv_x = trainX[sv_idx]
    sv_y = trainY[sv_idx]

    b_s = []
    for j in range(len(a)):
        gaus_res = gaussian_kernel(trainX[sv_idx], sv_x[j], sigma)
        eq = gaus_res * sv_y * a
        wtx = sv_y[j] - np.sum(eq)
        b_s.append(wtx)

    b = np.mean(b_s)

    predictions_train = np.sign(np.sum(a * sv_y * K[sv_idx][:, sv_idx], axis=1) + b)
    accuracy_train = np.mean(predictions_train == trainY[sv_idx])

    return a, sv_x, sv_y, b, accuracy_train


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

C_values = [1, 10, 100, 1000]
sigmaArr = [.001, .01, .1, 1, 10, 100]

for sigma in sigmaArr:
    for C in C_values:
        a, sv_x, sv_y, b, accuracy_train = train_svm(trainX, trainY, sigma, C)

        K_val = np.zeros((len(validationY), len(sv_y)))
        for i in range(len(validationY)):
            for j in range(len(sv_y)):
                K_val[i, j] = gaussian_kernel(validationX[i], sv_x[j], sigma)

        predictions_validation = np.sign(np.sum(a * sv_y * K_val, axis=1) + b)
        accuracy_validation = np.mean(predictions_validation == validationY)

        K_test = np.zeros((len(testY), len(sv_y)))
        for i in range(len(testY)):
            for j in range(len(sv_y)):
                K_test[i, j] = gaussian_kernel(testX[i], sv_x[j], sigma)

        predictions_test = np.sign(np.sum(a * sv_y * K_test, axis=1) + b)
        accuracy_test = np.mean(predictions_test == testY)

        print(f"C = {C}, sigma = {sigma}, Training Accuracy = {accuracy_train}, Validation Accuracy = {accuracy_validation}, Test Accuracy = {accuracy_test}")


