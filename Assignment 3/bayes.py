import numpy as np


def naive_bayes(data):
    prior = gradient_descent(data, [], np.array([10.] + [0.5]), bayes_gradient)
    param = build_bayes_param(data[0], prior[1:], prior[0])
    return param, prior


def conditional_probability(data, labels, p, m):
    p_fake = (data.T * labels + p * m) / (np.sum(labels) + m)
    not_labels = np.logical_not(labels)
    p_real = (data.T * not_labels + p * m) / (np.sum(not_labels) + m)
    return p_fake, p_real


def bayes_is_fake(param, data):
    return eval_bayes(param, data) > 0


def eval_bayes(param, data):
    return param[0] + data * param[1:]


def cross_entropy(yh, y):
    s = 1 / (1 + np.exp(-yh))
    return -np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))


def build_bayes_param(data, p, m):
    p_fake, p_real = conditional_probability(data[0], data[1], p, m)
    n = len(p_fake)
    param = np.empty(n + 1)
    param[0] = np.log(np.sum(data[1]).astype(float) / (n - np.sum(data[1]))) \
               + np.sum(np.log((1 - p_fake) / (1 - p_real)))
    param[1:] = np.log(p_fake / p_real) - np.log((1 - p_fake) / (1 - p_real))
    return param


def bayes_gradient(data, labels, p, h=1e-5):
    param = build_bayes_param(data[0], p[1:], p[0])
    y0 = cross_entropy(eval_bayes(param, data[1][0]), data[1][1])
    dp = np.empty_like(p)
    dparam = build_bayes_param(data[0], p[1:], p[0] + h)
    dp[0] = (cross_entropy(eval_bayes(dparam, data[1][0]), data[1][1]) - y0) / h
    for i in range(1, len(p)):
        p_mod = p.copy()
        p_mod[i] = p_mod[i] + h
        dparam = build_bayes_param(data[0], p_mod[1:], p_mod[0])
        dp[i] = (cross_entropy(eval_bayes(dparam, data[1][0]),
                               data[1][1]) - y0) / h
    return dp


def gradient_descent(data, labels, parameters, gradient, learning_rate=1e-5,
                     epsilon=4e-5, max_iter=1e5):
    parameters = parameters.copy()  # ensure that passed in value not changed
    last = np.zeros_like(parameters)
    i = 0
    while np.linalg.norm(parameters - last) > epsilon and i < max_iter:
        last = parameters.copy()
        parameters -= gradient(data, labels, parameters) \
                      * learning_rate
        i += 1
    return parameters
