import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1-x)
    return 1 / (1 + np.exp(-x))

X = np.array([
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
                ])


y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

weight_matrix0 = 2 * np.random.random((3, 1)) - 1

for iter in range(10000):

    guess0 = X
    guess1 = sigmoid(np.dot(guess0, weight_matrix0))

    guess1_error = y - guess1

    guess1_delta = guess1_error * sigmoid(guess1, True)

    weight_matrix0 += np.dot(guess0.T, guess1_delta)

print ("Ouput")
print (guess1)
