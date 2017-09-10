# Basic sigmoid function using math library and int as input
import math
import time
import numpy as np


def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s


print(basic_sigmoid(3))


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


# sigmoid_derivative

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


print("sigmoid_derivative(x) = " + str(sigmoid_derivative(a)))


def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    return v


image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])


print("image2vector(image) = " + str(image2vector(image)))


# comparing for loops to vectorization
a = np.random.rand(1000000)
b = np.random.rand(1000000)


tic = time.time()
c = np.dot(a, b)
toc = time.time()


print('c = ' + str(c) + ' vectorized version: ' + str(1000 * (toc - tic)) + 'ms')


c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()


print('c = ' + str(c) + ' for loop version: ' + str(1000 * (toc - tic)) + 'ms')

A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)

print(B.shape)
