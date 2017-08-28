# Basic sigmoid function using math library and int as input
import math


def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s


print(basic_sigmoid(3))


import numpy as np


a = np.array([1, 2, 3])
print(a)
print(np.exp(a))
print(a + 3)


# numpy sigmoid of vector/matrix
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


print(sigmoid(a))
print(a.shape)
