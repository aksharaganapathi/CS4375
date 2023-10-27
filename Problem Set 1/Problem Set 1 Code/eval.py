import numpy as np

w = [[-0.69607841], [24.76674531], [40.15149569], [41.22081426], [8.94285389], [11.53221129], [-16.33933404],
     [-10.26325461], [-5.94665761], [-13.10014128], [-11.2854318], [-17.26401287], [-13.32157989], [3.58849385],
     [-11.82224669], [1.76748428], [0.64142088], [-14.92656629], [25.14249461], [25.72386069], [26.33431068],
     [32.42237144], [-4.07120256], [14.73918214], [14.71248721], [18.37504251], [-31.59987655], [-24.63605925],
     [-1.06260087], [2.70821962]]

bias = 28.511283837809273


def eval(dataset):
    second_degree_features = [dataset[:, i] * dataset[:, j] for i in range(4) for j in range(i, 4)]
    third_degree_features = [dataset[:, i] * dataset[:, j] * dataset[:, k] for i in range(4) for j in range(i, 4) for k
                             in range(j, 4)]
    x = np.hstack((np.array(second_degree_features).T, np.array(third_degree_features).T))

    pred = np.sign(np.dot(x, w) + bias)
    return pred

with open('mystery.data', 'r', encoding='utf-8-sig') as file:
    data = file.read().splitlines()
    data = [line.split(',') for line in data]
    mystery_data = np.array(data, dtype=float)

X = mystery_data[:, :-1]
print(eval(X))