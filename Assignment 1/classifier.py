from __future__ import absolute_import, division, print_function

import numpy as np
from get_data import load_data


def cost_function(data, labels, parameters):
    cost = np.linalg.norm(np.dot(data, parameters[1:]) + parameters[0] - labels)
    return cost ** 2 / len(data)


def cost_gradient(data, labels, parameters):
    temp = np.dot(data, parameters[1:]) + parameters[0] - labels
    gradients = np.empty_like(parameters)
    gradients[0] = np.sum(temp)
    gradients[1:] = np.dot(data.T, temp)
    return 2 * gradients / len(data)


def labelling_accuracy(data, real_labels, parameters):
    # Note that by passing only a single image as part of data and one label,
    # the output of this function can be interpreted as a boolean signifying
    # whether the generated classifier believes that the given image is of the
    # actor siginified by the given label
    results = np.dot(data, parameters[1:]) + parameters[0]
    if len(parameters.shape) == 1: # Binary label, 0 or 1
        return np.sum((results > 0.5) == real_labels) / len(real_labels)
    else: # 3+ different sets to classify
        gen_labels = np.zeros_like(real_labels)
        for i, r in enumerate(results):
            gen_labels[i, np.argmax(r)] = 1
        return np.sum(np.all(gen_labels == real_labels, 1)) / len(real_labels)


def gradient_descent(data, labels, parameters, learning_rate=1e-3, epsilon=4e-5,
                     max_iter=1e5):
    parameters = parameters.copy()  # ensure that passed in value not changed
    last = np.zeros_like(parameters)
    i = 0
    while np.linalg.norm(parameters - last) > epsilon and i < max_iter:
        last = parameters.copy()
        parameters -= cost_gradient(data, labels, parameters) \
                      * learning_rate
        i += 1
    return parameters


def classify(names, set_sizes=(100, 10, 10), learning_rate=1e-3, epsilon=4e-5,
             max_iter=1e5, is_random=False):
    # ensure that each name set is formated as a list (allow for single entries)
    names = map(lambda x: [x] if isinstance(x, basestring) else x, names)

    # initialize the different sets
    total_entries = len(np.array(names).flatten())
    training = np.empty((total_entries * set_sizes[0], 1024),
                        dtype='float64')
    validation = np.empty((total_entries * set_sizes[1], 1024),
                          dtype='float64')
    testing = np.empty((total_entries * set_sizes[2], 1024),
                       dtype='float64')

    # load images into the three different sets
    for i, name in enumerate(np.array(names).flatten()):
        images = load_data(name, sizes=set_sizes, is_random=is_random)
        training[i * set_sizes[0]: i * set_sizes[0] + set_sizes[0]] = \
            images[:set_sizes[0]]
        validation[i * set_sizes[1]: i * set_sizes[1] + set_sizes[1]] = \
            images[set_sizes[0]:-set_sizes[2]]
        testing[i * set_sizes[2]: i * set_sizes[2] + set_sizes[2]] = \
            images[-set_sizes[2]:]

    # create the proper labels for the sets
    # note that since we do batch updating, the order of the different actors
    # does not matter, so all of one actor is first, then all of another, etc.
    if len(names) == 2:
        training_labels = np.hstack((np.ones(len(names[0]) * set_sizes[0]),
                                     np.zeros(len(names[1]) * set_sizes[0])))
        validation_labels = np.hstack((np.ones(len(names[0]) * set_sizes[1]),
                                       np.zeros(len(names[1]) * set_sizes[1])))
        testing_labels = np.hstack((np.ones(len(names[0]) * set_sizes[2]),
                                    np.zeros(len(names[1]) * set_sizes[2])))
        parameters = np.ones(1025) * 1e-4
    else:
        training_labels = np.zeros((total_entries * set_sizes[0], len(names)))
        validation_labels = np.zeros((total_entries * set_sizes[1], len(names)))
        testing_labels = np.zeros((total_entries * set_sizes[2], len(names)))
        i = 0
        for j, name_set in enumerate(names):
            training_labels[
            i * set_sizes[0]:(i + len(name_set)) * set_sizes[0], j] = 1
            validation_labels[
            i * set_sizes[1]:(i + len(name_set)) * set_sizes[1], j] = 1
            testing_labels[
            i * set_sizes[2]:(i + len(name_set)) * set_sizes[2], j] = 1
            i += len(name_set)
        parameters = np.ones((1025, len(names))) * 1e-4

    # perform gradient descent and evaluate the results
    parameters = gradient_descent(training, training_labels, parameters,
                                  learning_rate, epsilon, max_iter)
    cost = [cost_function(training, training_labels, parameters),
              cost_function(validation, validation_labels, parameters),
              cost_function(testing, testing_labels, parameters)]
    accuracy = [labelling_accuracy(training, training_labels, parameters),
                labelling_accuracy(validation, validation_labels, parameters),
                labelling_accuracy(testing, testing_labels, parameters)]

    return parameters, cost, accuracy
