import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict, compute_cost, initialize_parameters
from model import model

if __name__ == '__main__':
    np.random.seed(1)
    # load data sets
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    parameters = model(X_train, Y_train, X_test, Y_test)
