from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from time import time


def softmax(y):
    return (np.exp(y).T / np.sum(np.exp(y), 1)).T


def NLL(p, y):
    return -np.sum(y * np.log(p))


def basic_network(W, x, b):
    return softmax(np.dot(x, W) + b)


def basic_gradient(W, x, b, y):
    dp = basic_network(W, x, b) - y
    return np.dot(x.T, dp), np.sum(dp, 0)


def labelling_accuracy(W, x, b, y):
    results = basic_network(W, x, b)
    y_ = np.zeros_like(y)
    for i, r in enumerate(results):
        y_[i, np.argmax(r)] = 1
    return np.sum(np.all(y_ == y, 1)) / len(y)


def gradient_descent(W, x, b, y, xv, yv, learning_rate=1e-5, epsilon=1e-5,
                     max_iter=10000, momentum=0.9, save_file=None, save_int=1,
                     batch_size=1000, timeout=np.Inf):
    W = W.copy()  # ensure that passed in value not changed
    b = b.copy()
    last = np.zeros_like(W)
    i = 0
    training_perf = []
    validation_perf = []
    recorded_at = []
    dW, db = 0, 0
    next_record = 0
    start_time = time()
    while np.linalg.norm(W - last) > epsilon and i < max_iter and \
            time() - start_time < timeout:
        if i == next_record and xv and yv:
            training_perf.append(labelling_accuracy(W, x, b, y))
            validation_perf.append(labelling_accuracy(W, xv, b, yv))
            recorded_at.append(i)
            print("Iteration {}. Training Accuracy = {:f}. "
                  "Validation Accuracy = {:f}"
                  "".format(i, training_perf[-1], validation_perf[-1]))
            if i > 0:
                next_record += max(1, int((recorded_at[-1] - recorded_at[-2]) *
                    save_int / abs(training_perf[-1] - training_perf[-2]) / 10))
            else:
                next_record = 1

        last = W.copy()
        if batch_size:
            subset = np.random.choice(len(x), batch_size, replace=False)
            dW_n, db_n = basic_gradient(W, x[subset], b, y[subset])
        else:
            dW_n, db_n = basic_gradient(W, x, b, y)
        dW = dW * momentum - dW_n * learning_rate
        db = db * momentum - db_n * learning_rate
        W += dW
        b += db
        i += 1

    if xv and yv:
        training_perf.append(labelling_accuracy(W, x, b, y))
        validation_perf.append(labelling_accuracy(W, xv, b, yv))
        recorded_at.append(i)
        print("Iteration {}. Training Accuracy = {:f}. "
              "Validation Accuracy = {:f}"
              "".format(i, training_perf[-1], validation_perf[-1]))

    if save_file is not None:
        plt.plot(recorded_at, training_perf)
        plt.plot(recorded_at, validation_perf)
        plt.ylim((0.7, 1))
        plt.xlabel('Iterations')
        plt.ylabel('Normalized Error')
        plt.legend(('Training', 'Validation'))
        plt.savefig(save_file)
        plt.show()

    return W, b
