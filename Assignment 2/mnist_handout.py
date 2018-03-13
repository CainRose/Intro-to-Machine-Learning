from __future__ import absolute_import, division, print_function

import cPickle

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as imio
from scipy.io import loadmat

# Load the MNIST digit data
M = loadmat("mnist_all.mat")

# Display the 150-th "5" digit from the training set
imio.imshow(M["train5"][150].reshape((28, 28)), cmap=plt.cm.gray)
imio.show()


def softmax(y):
    """Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases"""
    return np.exp(y) / np.tile(np.sum(np.exp(y), 0), (len(y), 1))


def tanh_layer(y, W, b):
    """Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases"""
    return np.tanh(np.dot(W.T, y) + b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = np.dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


def NLL(y, y_):
    return -np.sum(y_ * np.log(y))


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 = y - y_
    dCdW1 = np.dot(L0, dCdL1.T)


# Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300, 1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10, 1))

# Load one example from the training set, and run it through the
# neural network
x = M["train5"][148:149].T
L0, L1, output = forward(x, W0, b0, W1, b1)
# get the index at which the output is the largest
y = np.argmax(output)

################################################################################
# Code for displaying a feature from the weight matrix mW
# fig = figure(1)
# ax = fig.gca()
# heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)
# fig.colorbar(heatmap, shrink = 0.5, aspect=5)
# show()
################################################################################
