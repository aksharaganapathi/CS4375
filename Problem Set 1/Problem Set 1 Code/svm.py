import numpy as np
import cvxopt

with open('mystery.data', 'r', encoding='utf-8-sig') as file:
    data = file.read().splitlines()
    data = [line.split(',') for line in data]
    mystery_data = np.array(data, dtype=float)

X = mystery_data[:, :-1]
Y = mystery_data[:, -1]

second_degree_features = [X[:, i] * X[:, j] for i in range(4) for j in range(i, 4)]
third_degree_features = [X[:, i] * X[:, j] * X[:, k] for i in range(4) for j in range(i, 4) for k in range(j, 4)]

X = np.hstack((np.array(second_degree_features).T, np.array(third_degree_features).T))

row, col = X.shape

P = cvxopt.matrix(np.eye(col + 1))
P[0, 0] = 0
q = cvxopt.matrix(np.zeros((col + 1, 1)))
G = cvxopt.matrix(-Y.reshape(-1, 1) * np.hstack((X, np.ones((row, 1)))))
h = cvxopt.matrix(-np.ones((row, 1)))

output = cvxopt.solvers.qp(P, q, G, h)

w = np.array(output['x'][1:])
bias = np.array(output['x'][0])
support_vectors = np.where(np.abs(np.dot(X, w) + bias) - 1 < 1e-10)[0]
print("Support Vectors: ", support_vectors)
print("Margin:", 1 / np.linalg.norm(w))
print("Weights:", w)
print("Bias:", bias)



