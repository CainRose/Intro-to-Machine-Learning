import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import classifiers
import data_processing

dtype_float = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


# def get_error(model, loss_fn, data):
#     return loss_fn(model(data[0]), data[1]).data.cpu().numpy()


def get_error(model, loss_fn, dataset):
    dataloader = data_processing.Loader(dataset, 100)
    loss = 0
    for sample in dataloader:
        x, y = sample
        x = Variable(x.type(dtype_float), requires_grad=False)
        y = Variable(y.type(dtype_long), requires_grad=False)
        y_pred = model(x)
        loss += loss_fn(y_pred, y).data.cpu().numpy()[0]
    return loss / len(dataset) * 100


def get_accuracy(model, dataset):
    data_loader = data_processing.Loader(dataset, 100)
    acc = 0
    model = model.cpu()
    for sample in data_loader:
        x, y = sample
        x = Variable(x.type(dtype_float), requires_grad=False).cpu()
        y = y.cpu().numpy()
        y_pred = np.argmax(model(x).cpu().data.numpy(), 1)
        acc += np.sum(y_pred == y)
    return acc / len(dataset)


def train_classifier(model, loss_fn, training_set, validation_set,
                     batch_size=50, epochs=100, l2=3, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_loader = data_processing.Loader(training_set, 50)
    patience_count = 0
    best_loss = np.inf
    model.eval()
    training_loss = [get_error(model, loss_fn, training_set)]
    validation_loss = [get_error(model, loss_fn, validation_set)]
    model.train()
    print('Initial\t\tTraining Error: {:f}    Valdiation Error: {:f}'.format(
        training_loss[0], validation_loss[0]))
    for i in range(epochs):
        print("Beginning Epoch {:3}.".format(i), end='    ')
        training_loss.append(0)
        model.train()
        for sample in training_loader:
            model.zero_grad()
            x, y = sample
            x = Variable(x.type(dtype_float), requires_grad=False)
            y = Variable(y.type(dtype_long), requires_grad=False)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            training_loss[-1] += loss.data.cpu().numpy()[0]
            loss.backward()
            optimizer.step()
        training_loss[-1] /= len(training_set) / 50
        model.eval()
        validation_loss.append(get_error(model, loss_fn, validation_set))
        print('Training Error: {:f}    Valdiation Error: {:f}'.format(
            training_loss[-1], validation_loss[-1]))
        if validation_loss[-1] < best_loss:
            patience_count = 0
            best_loss = validation_loss[-1]
        else:
            patience_count += 1
            if patience_count == patience:
                break
    model.eval()
    return training_loss, validation_loss
