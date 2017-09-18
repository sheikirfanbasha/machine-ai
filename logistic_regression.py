import numpy as np
import time
from utils import load_data_set, sigma

tic = time.time()

# Load the data required for training and test
train_X_orig, train_y_orig, test_X_orig, test_y_orig = load_data_set()

# Make sure they are of the correct shape
print "Original data shape"
print train_X_orig.shape
print train_y_orig.shape
print test_X_orig.shape
print test_y_orig.shape

# Flatten the train and test data features from 3D to 1D
train_X = train_X_orig.reshape(train_X_orig.shape[1] * train_X_orig.shape[2] * train_X_orig.shape[3], train_X_orig.shape[0])
test_X = test_X_orig.reshape(test_X_orig.shape[1] * test_X_orig.shape[2] * test_X_orig.shape[3], test_X_orig.shape[0])


# Normalization
# This step is crucial. If avoided then huge values will be passed on to processing which may lead to overflow errors
# particularly in this case. We would see "overflow encountered in exp"
train_X = train_X/255
test_X = test_X/255

# Ensure the data is flatten to the right shape
# print "Data shape after flattening"
print train_X.shape
# print test_X.shape

# Reshape the targets
train_y_orig = train_y_orig.reshape(1, train_y_orig.shape[0])
test_y_orig = test_y_orig.reshape(1, test_y_orig.shape[0])

# Initialize the weights and bias
W = np.zeros((train_X.shape[0], 1))
b = 0  # will be scaled as per broadcasting

# Ensure the shape of w and b are right
# print "Shape of w"
print W.shape

# Define the number of iterations and learning rate values
N = 100
learning_rate = 0.00005
M = float(train_X.shape[1])
print M
# print M
Y = train_y_orig.T

# For each iteration do the training

for i in range(N):
    # Compute the output z = wx + b
    Z = np.dot(W.T, train_X) + b
    # Since we are doing logistic regression, compute the sigma of the output z
    A = sigma(Z)
    # Print the shape of the output y_hat computed
    print M
    cost = (- 1 / M) * (np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A)))
    # Print the cost computed
    print ("Cost after iteration %d is %f" % (i, cost))
    # Compute the change to be done for the weights and change to be done for bias
    dw = (1 / M) * np.dot(train_X, (A - Y).T)
    db = (1 / M) * np.sum((A - Y))

    W = W - learning_rate * dw

    b = b - learning_rate * db
    #print W[1][1]
    # print b

toc = time.time()

print("Time taken: %s", str(1000 * (toc - tic)) + "ms")
