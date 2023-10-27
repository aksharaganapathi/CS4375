import numpy as np

data = np.genfromtxt("perceptron.data", delimiter=',')

X = data[:, :-1]  # Gets all rows and all columns except the last one
y = data[:, -1]  # Gets all rows and the last column

# X has a shape of (n, m) where n is the number of rows and m is the number of columns.
# Returns 4 in this case since the shape is (numRows, 4)
w = np.zeros(X.shape[1])
b = 0

maxIterations = 10000
iterationCount = 0

converged = False
for iteration in range(maxIterations):
    misclassified = False
    wSum = 0
    bSum = 0

    for i in range(len(X)):
        if y[i] * (np.dot(X[i], w) + b) <= 0:
            wSum += y[i] * X[i]
            bSum += y[i]
            misclassified = True

    if not misclassified:
        print("Perfect classifier found after", iterationCount + 1, "iterations")
        converged = True
        break
    else:
        w += wSum
        b += bSum
        iterationCount += 1

    if iteration < 3:
        print("Iteration", iteration + 1)
        print("w:", w)
        print("b:", b)


if not converged:
    print("Algorithm did not converge after", maxIterations, "iterations")

# Print the final results
print("Final weights:", w)
print("Final bias:", b)
