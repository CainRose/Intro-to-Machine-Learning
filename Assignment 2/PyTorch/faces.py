import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
from torch.autograd import Variable
from get_data import load_data

dtype_float = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


def weight_init(model):
    std_dev = 2.0 / (model[0].in_features + model[-1].out_features)
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data.normal_(0.0, std_dev)


def select_data(actors, imagesize=32):
    sizes = [(54, 16, 16)] + [(64, 20, 20)] * 5

    training = []
    validation = []
    testing = []
    train_labels = []
    valid_labels = []
    test_labels = []

    l = np.array([1] + [0] * 5)
    for i in range(6):
        d = load_data(actors[i], sum(sizes[i]), 'bw' + str(imagesize))
        training.append(d[:sizes[i][0]])
        validation.append(d[sizes[i][0]:-sizes[i][2]])
        testing.append(d[-sizes[i][2]:])
        train_labels.append(np.ones(sizes[i][0]) * i)
        valid_labels.append(np.ones(sizes[i][1]) * i)
        test_labels.append(np.ones(sizes[i][2]) * i)

    training = Variable(torch.from_numpy(np.vstack(training))
                        .type(dtype_float), requires_grad=False)
    validation = Variable(torch.from_numpy(np.vstack(validation))
                          .type(dtype_float), requires_grad=False)
    testing = Variable(torch.from_numpy(np.vstack(testing))
                       .type(dtype_float), requires_grad=False)
    train_labels = Variable(torch.from_numpy(np.hstack(train_labels))
                            .type(dtype_long), requires_grad=False)
    valid_labels = Variable(torch.from_numpy(np.hstack(valid_labels))
                            .type(dtype_long), requires_grad=False)
    test_labels = Variable(torch.from_numpy(np.hstack(test_labels))
                           .type(dtype_long), requires_grad=False)

    return (training, train_labels), (validation, valid_labels), \
           (testing, test_labels)


def get_error(model, loss_fn, data):
    return loss_fn(model(data[0]), data[1]).data.cpu().numpy()


def train_classifier(model, loss_fn, train, valid, learning_rate=1e-3,
                     batch_size=10, iterations=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_error = []
    valid_error = []
    for i in range(iterations):
        if not i % 10:
            train_error.append(get_error(model, loss_fn, train))
            valid_error.append(get_error(model, loss_fn, valid))
            print(i)

        idx = np.random.permutation(len(train[0]))
        for j in range(0, len(train[0]), batch_size):
            x = train[0][idx[j:j + batch_size],]
            y = train[1][idx[j:j + batch_size],]

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            model.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        train_error.append(get_error(model, loss_fn, train))
        valid_error.append(get_error(model, loss_fn, valid))
    return train_error, valid_error


def get_model(image_size, hidden, activation):
    return torch.nn.Sequential(
        torch.nn.Linear(image_size ** 2, hidden),
        activation,
        torch.nn.Linear(hidden, 6),
    ).cuda()


def test_heperparameters(act, img, hid, itr, bat, atv, lrn):
    seed = 10007
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, valid, test = select_data(act, img)
    model = get_model(img, hid, atv)
    weight_init(model)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    train_classifier(model, loss_fn, train, valid, lrn, bat, itr)
    actual = valid[1].data.cpu().numpy()
    prediction = model(valid[0]).data.cpu().numpy()
    return np.mean(np.argmax(prediction, 1) == actual)


def plot_performance(act, img, hid, itr, bat, atv, lrn, save=False):
    seed = 10007
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, valid, test = select_data(act, img)
    model = get_model(img, hid, atv)
    weight_init(model)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    train_error, valid_error = \
        train_classifier(model, loss_fn, train, valid, lrn, bat, itr)

    plt.plot(np.arange(0, len(train_error)) * 10, train_error)
    plt.plot(np.arange(0, len(valid_error)) * 10, valid_error)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(('Training Set', 'Validation Set'))
    if save:
        plt.savefig('q8.png')
    plt.show()
    actual = test[1].data.cpu().numpy()
    prediction = model(test[0]).data.cpu().numpy()
    print('Testing Set Error:', get_error(model, loss_fn, test)[0])
    print('Testing Set Accuracy:', np.mean(np.argmax(prediction, 1) == actual))
    return model


def grid_search():
    act = ['gilpin', 'bracco', 'harmon', 'baldwin', 'hader', 'carell']

    # Grid-based search initialization
    image_size = [32, 64]
    hidden_sizes = [10, 20, 50]
    epochs = [200, 300, 500]
    batch_size = [20, 50, 100]
    activation = [torch.nn.ReLU(), torch.nn.LeakyReLU()]
    learning = [5e-4, 1e-3]
    combinations = itertools.product(image_size, hidden_sizes, epochs,
                                     batch_size, activation, learning)

    best_score = 0
    best = ()
    for img, hid, epo, bat, atv, lrn in combinations:
        print('Testing the following parameters:')
        print('\tImage Size:', img)
        print('\tHidden Layer Size:', hid)
        print('\tEpochs:', epo)
        print('\tBatch Size:', bat)
        print('\tActivation Function:', atv)
        print('\tInitial Learning Rate:', lrn)
        performance = test_heperparameters(act, img, hid, epo, bat, atv, lrn)
        if performance > best_score:
            best_score = performance
            best = (img, hid, epo, bat, atv, lrn)
        print(' ->\tPerformace:', performance)
        print('\tCur Best:', best_score, best)

    img, hid, epo, bat, atv, lrn = best
    with open('q8best.txt', 'w') as f:
        f.write('Image Size: {0}\n'.format(str(img)))
        f.write('Hidden Layer Size: {0}\n'.format(str(hid)))
        f.write('Iterations: {0}\n'.format(str(epo)))
        f.write('Batch Size: {0}\n'.format(str(bat)))
        f.write('Activation Function: {0}\n'.format(str(atv)))
        f.write('Initial Learning Rate: {0}\n'.format(str(lrn)))

    print('Finished. The best parameters are:')
    print('\tImage Size:', img)
    print('\tHidden Layer Size:', hid)
    print('\tEpochs:', epo)
    print('\tBatch Size:', bat)
    print('\tActivation Function:', atv)
    print('\tInitial Learning Rate:', lrn)
    print('\n')
    plot_performance(act, img, hid, epo, bat, atv, lrn, True)


def main():
    print("Please select question to display output for:\n" +
          "\t8 9")
    question = input()
    if question == '8':
        grid_search()
    elif question == '9':
        act = ['gilpin', 'bracco', 'harmon', 'baldwin', 'hader', 'carell']
        model = plot_performance(act, 64, 20, 200, 100,
                                 torch.nn.LeakyReLU(), 0.001)
        # Model up to the hidden layer
        hidden = torch.nn.Sequential(model[0], model[1])
        # Images specific to each actor
        harmon = Variable(torch.from_numpy(load_data('harmon', 100, 'bw64'))
                          .type(dtype_float))
        baldwin = Variable(torch.from_numpy(load_data('baldwin', 100, 'bw64'))
                           .type(dtype_float))
        # Get the activations and signifigance of each hidden layer
        harmon_activ = np.sum(hidden(harmon).data.cpu().numpy(), 0)
        baldwin_activ = np.sum(hidden(baldwin).data.cpu().numpy(), 0)
        harmon_last = model[2].weight.data[2].cpu().numpy()
        baldwin_last = model[2].weight.data[3].cpu().numpy()
        harmon_signif = harmon_activ * harmon_last
        baldwin_signif = baldwin_activ * baldwin_last
        # Find differences
        diffs = np.abs(harmon_signif - baldwin_signif)
        most_diff = np.argpartition(diffs, -4)[-4:][::-1]

        plt.figure(figsize=(5, 5))
        for i, idx in enumerate(most_diff):
            plt.subplot(2, 2, i + 1)
            weights = model[0].weight.data[i].cpu().numpy()
            plt.imshow(weights.reshape((64, 64)), cmap=plt.cm.coolwarm)
            if harmon_signif[i] > baldwin_signif[i]:
                plt.title('Angie Harmon')
            else:
                plt.title('Alec Baldwin')
        plt.tight_layout()
        plt.savefig('q9.png')
        plt.show()


if __name__ == '__main__':
    main()
