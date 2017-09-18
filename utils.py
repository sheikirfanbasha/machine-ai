import h5py
import numpy as np


def load_data_set():
    """Read the dataset from file and populates the respective variables"""
    # train_data.keys() will give all the keys it has like "train_set_x"
    train_data = h5py.File("datasets/train_catvnoncat.h5")
    train_X = train_data["train_set_x"][:]
    train_Y = train_data["train_set_y"][:]
    test_data = h5py.File("datasets/test_catvnoncat.h5")
    test_X = test_data["test_set_x"][:]
    test_Y = test_data["test_set_y"][:]
    return train_X, train_Y, test_X, test_Y


def sigma(x):
    return 1 / (1 + np.exp(-x))


