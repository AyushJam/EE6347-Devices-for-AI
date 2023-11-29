# Train the Logistic Regression Algorithm

import numpy as np
import matplotlib.pyplot as plt
import csv
from lr_utils import (
    load_dataset,
    convert_to_grayscale,
    get_feature_vector,
    sigmoid,
    calculate_cost,
)

train_set_x, train_set_y, test_x_og, test_y_og, classes = load_dataset()
# print(train_set_x.shape) # 209 64 64 3
# print(train_set_y.shape) # 1 209
# print(test_x.shape) # 209 64 64 3
# print(classes) # notcat cat


def save_model(weights, b, file_name):
    # b is a scalar and weights is an array of dimensions 1 x 12288
    # store these in a file
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([b])
        for w in weights[0]:
            writer.writerow([w])


def compute_gradient_of_cost_function(X, Y, W, b):
    Z = (np.matmul(W, X.T) + b).T
    Y_hat = sigmoid(Z)

    # write_to_file("Y_hat.txt", Y_hat)

    dB = (np.sum(Y_hat - Y)) / m
    dW = ((np.matmul(X.T, (Y_hat - Y))) / m).T
    # print(dW.shape, dB.shape)
    return dW, dB


def optimize_weights_using_gradient_descent(X, Y, W, b, threshold, learning_rate):
    prev_cost = 0
    iteration_num = 0
    # print(W.shape)
    while True:
        iteration_num += 1
        dW, dB = compute_gradient_of_cost_function(X, Y, W, b)
        # print(W.shape)
        W = W - learning_rate * dW
        b = b - learning_rate * dB

        # print w and dw shapes
        # print(W.shape, dW.shape)
        Y_hat = sigmoid((np.matmul(W, X.T) + b).T)
        # print(Y.shape)
        cost = calculate_cost(Y, Y_hat)

        if abs(prev_cost - cost) <= threshold:
            print(iteration_num, cost)
            break

        # if iteration_num % 1e1 == 0:
        #     print(iteration_num, cost)

        print(iteration_num, cost)
        # print("W: ", W.shape, "b: ", b.shape)
        prev_cost = cost.copy()
    # plot_cost(costs, plot_x)
    return W, b


X = get_feature_vector(train_set_x)
m = X.shape[0]  # 209 * 12288

Y = train_set_y.T  # 209 * 1

# Initialize weights
W = np.random.rand(1, 64 * 64 * 3) * 1e-3
b = 0

learning_rate = 0.005
threshold = 2e-5

W, b = optimize_weights_using_gradient_descent(X, Y, W, b, threshold, learning_rate)

# save weights to file
save_model(W, b, "weights.csv")

"""
Threshold    Iterations   train accuracy   test accuracy
1e-3         289           0.84             0.58
1e-4         289           0.86             0.6
2e-5         5k            1.00             0.68
1e-5         5k            1.00             0.68
"""
