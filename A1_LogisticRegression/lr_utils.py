import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File("train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("test_catvnoncat.h5", "r")
    # print(type(test_dataset))
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_grayscale(data_rgb):
    data_gscale = np.sum(data_rgb[:][:][:])
    return data_gscale


def get_feature_vector(x_rgb_matrix):
    x = np.zeros((1, 64 * 64 * 3))
    for sample in x_rgb_matrix[:]:
        x = np.vstack((x, sample.reshape(1, 64 * 64 * 3)))
    x = x[1:] / 255
    return x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_cost(Y, Y_hat):
    nsamples = np.shape(Y)[0]
    # raise exception if Y hat is zero
    cost = -np.sum((Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))) / nsamples
    return cost
