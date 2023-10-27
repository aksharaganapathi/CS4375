import numpy as np

data = np.genfromtxt("perceptron.data", delimiter=',')

X = data[:, :-1]  # Gets all rows and all columns except the last one
y = data[:, -1]  # Gets all rows and the last column

# X has a shape of (n, m) where n is the number of rows and m is the number of columns.
# Returns 4 in this case since the shape is (numRows, 4)
w = np.zeros(X.shape[1])
b = 0

maxIterations = 1000000
iterationCount = 0
i = 0
converged = False


def checkWandB(X, y, w, b):
    error = 0
    for i in range(X.shape[0]):
        if np.sign(np.dot(w, X[i]) + b) != np.sign(y[i]):
            error += 1

    return error == 0


for iteration in range(maxIterations):
    misclassified = False

    if y[i] * (np.dot(X[i], w) + b) <= 0:
        w += y[i] * X[i]
        b += y[i]
        misclassified = True

    iterationCount += 1

    if iteration < 3:
        print("Iteration", iteration + 1)
        print("w:", w)
        print("b:", b)

    if checkWandB(X, y, w, b):
        print("Perfect classifier found after", iterationCount, "iterations")
        converged = True
        break

    if i == len(X) - 1:
        i = 0
    else:
        i += 1

if not converged:
    print("Algorithm did not converge after", maxIterations, "iterations")

# Print the final results
print("Final weights:", w)
print("Final bias:", b)
