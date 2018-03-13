import torch
import numpy as np
from scipy.optimize import minimize_scalar


def build_model(num_words):
    return torch.nn.Sequential(
        torch.nn.Linear(num_words, 2)
    )


def get_error(model, loss_fn, data):
    return loss_fn(model(data[0]), data[1]).data.cpu().numpy()


def train_classifier(model, loss_fn, train, valid, learning_rate=1e-3,
                     batch_size=10, iterations=10000, regularization=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_error = []
    valid_error = []
    for i in range(iterations):
        if not i % 10:
            train_error.append(get_error(model, loss_fn, train))
            valid_error.append(get_error(model, loss_fn, valid))
            # print(i)

        idx = np.random.permutation(len(train[0]))
        for j in range(0, len(train[0]), batch_size):
            x = train[0][idx[j:j + batch_size],]
            y = train[1][idx[j:j + batch_size],]

            y_pred = model(x)
            loss = loss_fn(y_pred, y) - \
                   float(regularization) * torch.norm(model[0].weight)

            model.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        train_error.append(get_error(model, loss_fn, train))
        valid_error.append(get_error(model, loss_fn, valid))
    return train_error, valid_error


def tune_regularization(data, iterations=500, batch_size=1000):
    def parameter_performance(param, train, valid, iterations, batch_size):
        model = build_model(train[0].shape[1]).cuda()
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        train_classifier(model, loss_fn, train, valid, iterations=iterations,
                         batch_size=batch_size, regularization=param)
        actual = valid[1].data.cpu().numpy()
        prediction = model(valid[0]).data.cpu().numpy()
        val = np.mean(np.argmax(prediction, 1) == actual)
        err = get_error(model, loss_fn, valid)[0]
        print('{:f} {:f}'.format(val, err))
        return err

    x = minimize_scalar(parameter_performance,
                        args=(data[0], data[1], iterations, batch_size))
    return x.x
