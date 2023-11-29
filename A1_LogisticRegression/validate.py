import numpy as np
import sklearn


def get_feature_vector(x_rgb_matrix):
    x = np.zeros((1, 64 * 64 * 3))
    for sample in x_rgb_matrix[:]:
        x = np.vstack((x, sample.reshape(1, 64 * 64 * 3)))
    x = x[1:] / 255
    return x


def check_weighted_f1_score(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.genfromtxt(predicted_test_Y_file_path, delimiter=",", dtype=np.int32)
    actual_Y = np.genfromtxt(actual_test_Y_file_path, delimiter=",", dtype=np.int32)
    from sklearn.metrics import f1_score

    weighted_f1_score = f1_score(actual_Y, pred_Y, average="weighted")
    # print("Weighted F1 score", weighted_f1_score)
    return weighted_f1_score


# check accuracy
def check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.genfromtxt(predicted_test_Y_file_path, delimiter=",", dtype=np.int32)
    actual_Y = np.genfromtxt(actual_test_Y_file_path, delimiter=",", dtype=np.int32)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(actual_Y, pred_Y)
    # print("Accuracy", accuracy)
    return accuracy
