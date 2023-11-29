import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset, sigmoid, get_feature_vector
from validate import check_weighted_f1_score, check_accuracy

# load w anb b from file
weights = np.genfromtxt("weights.csv", delimiter=",", dtype=np.float64)
b = weights[0]
W = weights[1:].reshape(1, 64 * 64 * 3)

train_set_x, train_set_y, test_x_og, test_y_og, classes = load_dataset()


# # 1. Test data
def test_data():
    np.savetxt("test_y_og.csv", test_y_og.T, delimiter=",", fmt="%d")
    test_x = get_feature_vector(test_x_og)
    predicted_test_y = np.where(sigmoid((np.matmul(W, test_x.T) + b).T) > 0.5, 1, 0)
    np.savetxt("predicted_test_y.csv", predicted_test_y, delimiter=",", fmt="%d")
    f1_score_test_data = check_weighted_f1_score(
        "test_y_og.csv", "predicted_test_y.csv"
    )
    accuracy_test_data = check_accuracy("test_y_og.csv", "predicted_test_y.csv")

    # print("F1 score for test data", round(f1_score_test_data, 2))
    print("Accuracy for test data", round(accuracy_test_data, 2))


# 2. Train data
def train_data():
    np.savetxt("train_y_og.csv", train_set_y.T, delimiter=",", fmt="%d")
    train_x = get_feature_vector(train_set_x)
    predicted_train_y = np.where(sigmoid((np.matmul(W, train_x.T) + b).T) > 0.5, 1, 0)
    np.savetxt("predicted_train_y.csv", predicted_train_y, delimiter=",", fmt="%d")
    f1_score_train_data = check_weighted_f1_score(
        "train_set_y.csv", "predicted_train_y.csv"
    )
    accuracy_train_data = check_accuracy("train_set_y.csv", "predicted_train_y.csv")

    # print("F1 score for train data", round(f1_score_train_data, 2))
    print("Accuracy for train data", round(accuracy_train_data, 2))


train_data()
test_data()

'''
# see if it can classify an inverted image from the test set
# 15 index is a cat. We will not rotate the image by 90 degrees
# and see if it can classify it correctly

# we will find all the indices of cats from the train set
# and then rotate them by 90 degrees and see if it can classify them correctly


def check_rotation(train_set_x, train_set_y):
    # find all the indices of cats from the train set
    cat_indices = np.where(train_set_y == 1)[1]

    # rotate all the cat images by 90 degrees
    # and see if it can classify them correctly
    cat_images = train_set_x[cat_indices]
    for image in cat_images:
        np.rot90(image, 1)

    # get the feature vector
    cat_images = get_feature_vector(cat_images)

    # now classify
    predicted_cat_images = np.where(
        sigmoid((np.matmul(W, cat_images.T) + b).T) > 0.5, 1, 0
    )

    # check accuracy
    accuracy = np.sum(predicted_cat_images) / len(cat_indices)
    print("Accuracy for rotated images", accuracy)


# check_rotation(train_set_x, train_set_y)


# similarly, test the model for horizontally stretched images
def check_stretch(train_set_x, train_set_y):
    # find all the indices of cats from the train set
    cat_indices = np.where(train_set_y == 1)[1]

    # stretch all the cat images horizontally
    # and see if it can classify them correctly
    cat_images = train_set_x[cat_indices]
    for image in cat_images:
        image = np.hstack((image, image))

    # get the feature vector
    cat_images = get_feature_vector(cat_images)

    # now classify
    predicted_cat_images = np.where(
        sigmoid((np.matmul(W, cat_images.T) + b).T) > 0.5, 1, 0
    )

    # check accuracy
    accuracy = np.sum(predicted_cat_images) / len(cat_indices)
    print("Accuracy for stretched images", accuracy)


# check_stretch(train_set_x, train_set_y)


# crop the image to 10 pixels and pad it to 64x64. then run the classification
# check the accuracy
def check_crop(train_set_x, train_set_y):
    # find all the indices of cats from the train set
    cat_indices = np.where(train_set_y == 1)[1]

    # crop all the cat images
    # and see if it can classify them correctly
    cat_images = train_set_x[cat_indices]
    for image in cat_images:
        image = image[0:10, 0:10]
        image = np.pad(image, ((0, 54), (0, 54), (0, 0)), "constant")

    # get the feature vector
    cat_images = get_feature_vector(cat_images)

    # now classify
    predicted_cat_images = np.where(
        sigmoid((np.matmul(W, cat_images.T) + b).T) > 0.5, 1, 0
    )

    # check accuracy
    accuracy = np.sum(predicted_cat_images) / len(cat_indices)
    print("Accuracy for cropped images", accuracy)


# check_crop(train_set_x, train_set_y)
'''