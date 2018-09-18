#!/usr/bin/python3

import numpy as np
np.random.seed(10)
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 1, 1, 0]).reshape(4, -1)

a1 = x
theta1 = np.random.random(12).reshape(4, 3) * 2 - 1
theta2 = np.random.random(4).reshape(1, 4) * 2 - 1

learning_rate = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(100000):
    # forward propagation: including the first layer and the second layer
    z1 = a1.dot(theta1.T)
    a2 = sigmoid(z1) 
    z2 = a2.dot(theta2.T)
    a3 = sigmoid(z2) 

    # backpropagation: chain rules of the partial derivatives
    a3_error = y - a3
    theta2_grad = np.dot(np.transpose(a3_error * (a3 * (1 - a3))), a2) 
    theta1_grad = np.dot(np.transpose(np.dot(a3_error * (a3 * (1 - a3)),  theta2) * (a2 * (1 - a2))), a1)

    # update the parameters until the gradient converges to 0
    theta2 += learning_rate * theta2_grad
    theta1 += learning_rate * theta1_grad

print('theta 1 printin :- ',theta1)
print('theta 2 printin :- ',theta2)
print('a3 printing  :- ',a3)
