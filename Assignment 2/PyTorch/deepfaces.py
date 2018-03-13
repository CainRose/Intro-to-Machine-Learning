import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.misc import imread
from skimage.color import gray2rgb
from torch.autograd import Variable

from faces import train_classifier, get_error, weight_init

dtype_float = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


# Provided class
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def get_activations(images):
    # Return the activations associated with the conv4 layer
    model = MyAlexNet()
    model.eval()
    # Neural network up to the fourth layer
    conv4 = torch.nn.Sequential(
        model.features[0], model.features[1], model.features[2],  # Conv1
        model.features[3], model.features[4], model.features[5],  # Conv2
        model.features[6], model.features[7],  # Conv3
        model.features[8], model.features[9],  # Conv4
    )
    # Calculate and flatten activations
    activations = conv4(images.cpu()).view(len(images), -1)
    return activations.cuda()


def load_data(name, size=None):
    np.random.seed(10007)
    # load 90 random images
    image_paths = glob.glob('alexnet/' + name + '[0-9]*.png')
    if size is not None:
        selected_paths = np.random.choice(image_paths, size, replace=False)
    else:
        selected_paths = image_paths
    images = []
    for path in selected_paths:
        try:
            im = imread(path)[:, :, :3]
        except IndexError:
            im = gray2rgb(imread(path))
        im = im - np.mean(im.flatten())
        im = im / np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(np.float32)
        images.append(im)
    return np.array(images)


def get_sets():
    # Get and format the actor data properly into sets
    act = ['gilpin', 'bracco', 'harmon', 'baldwin', 'hader', 'carell']
    sizes = [(54, 16, 16)] + [(64, 20, 20)] * 5

    training = []
    validation = []
    testing = []
    train_labels = []
    valid_labels = []
    test_labels = []

    l = np.array([1] + [0] * 5)
    for i in range(6):
        d = load_data(act[i], sum(sizes[i]))
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
    print('Data Loaded')

    training = get_activations(training)
    validation = get_activations(validation)
    testing = get_activations(testing)
    # Detach the values from AlexNet
    training = Variable(training.data, requires_grad=True)
    validation = Variable(validation.data, requires_grad=True)
    testing = Variable(testing.data, requires_grad=True)

    print('Activations Generated')

    return (training, train_labels), (validation, valid_labels), \
           (testing, test_labels)


def main():
    train, valid, test = get_sets()
    model = torch.nn.Sequential(
        torch.nn.Linear(65536, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 6),
    ).cuda()
    weight_init(model)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    train_error, valid_error = train_classifier(model, loss_fn, train, valid,
                                                iterations=100, batch_size=100)

    plt.plot(np.arange(0, len(train_error)) * 10, train_error)
    plt.plot(np.arange(0, len(valid_error)) * 10, valid_error)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(('Training Set', 'Validation Set'))
    plt.savefig('q10.png')
    plt.show()

    actual = test[1].data.cpu().numpy()
    prediction = model(test[0]).data.cpu().numpy()
    print('Testing Set Error:', get_error(model, loss_fn, test)[0])
    print('Testing Set Accuracy:', np.mean(np.argmax(prediction, 1) == actual))

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
