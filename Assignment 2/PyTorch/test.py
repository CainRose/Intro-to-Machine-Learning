from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

M = loadmat("mnist_all.mat")


def get_test(M):
    batch_xs = np.zeros((0, 28 * 28))
    batch_y_s = np.zeros((0, 10))

    test_k = ["test" + str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:]) / 255.)))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack(
            (batch_y_s, np.tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = np.zeros((0, 28 * 28))
    batch_y_s = np.zeros((0, 10))

    train_k = ["train" + str(i) for i in range(10)]
    for k in range(10):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:]) / 255.)))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack(
            (batch_y_s, np.tile(one_hot, (len(M[train_k[k]]), 1))))
    return batch_xs, batch_y_s


train_x, train_y = get_train(M)
test_x, test_y = get_test(M)

train_x, train_y = get_train(M)
test_x, test_y = get_test(M)

dim_x = 28 * 28
dim_h = 20
dim_out = 10

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

################################################################################
# Subsample the training set for faster training

train_idx = np.random.permutation(range(train_x.shape[0]))[:1000]
x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(
    dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)),
                     requires_grad=False).type(dtype_long)
#################################################################################


model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)

    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()  # Compute the gradient
    optimizer.step()  # Use the gradient information to
    # make a step

x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

y_pred = model(x).data.numpy()

model[0].weight.data.numpy()[10, :].shape

