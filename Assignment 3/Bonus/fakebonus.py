import matplotlib.pyplot as plt
import numpy as np
import torch

import classifiers
import data_processing
import train


def train_original():
    fake, real = data_processing.load_data()
    data, keywords = data_processing.process_data(fake, real)
    training_set = data_processing.Headlines(data[0])
    validation_set = data_processing.Headlines(data[1])
    testing_set = data_processing.Headlines(data[2])
    print('Data Loaded')
    model = classifiers.ConvnetClassifier(len(keywords),
                                          data[0][0][0].shape[1]).cuda()
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    training_loss, validation_loss = train.train_classifier(
        model, loss_fn, training_set, validation_set, patience=3)
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(('Training Set', 'Validation Set'))
    plt.savefig('error_orig.png')
    plt.show()
    torch.save(model.state_dict(), 'model_orig.pkl')
    model.eval()
    print('Acheived {:%} accuracy on the training set.'
          .format(train.get_accuracy(model, training_set)))
    print('Acheived {:%} accuracy on the validation set.'
          .format(train.get_accuracy(model, validation_set)))
    print('Acheived {:%} accuracy on the testing set.'
          .format(train.get_accuracy(model, testing_set)))


def train_expanded():
    train_data = data_processing.load_tsv('train.tsv')
    valid_data = data_processing.load_tsv('valid.tsv')
    test_data = data_processing.load_tsv('test.tsv')
    keywords = data_processing.generate_keywords(
        train_data[0], valid_data[0], test_data[0], save=True)
    print('Data Loaded')

    train_encoded = data_processing.encode_data(keywords, train_data[0], 40)
    training_set = data_processing.Headlines((train_encoded, train_data[1]))
    print('Training Set Processed')
    valid_encoded = data_processing.encode_data(keywords, valid_data[0], 40)
    validation_set = data_processing.Headlines((valid_encoded, valid_data[1]))
    print('Validation Set Processed')
    test_encoded = data_processing.encode_data(keywords, test_data[0], 40)
    testing_set = data_processing.Headlines((test_encoded, test_data[1]))
    print('Testing Set Processed')

    model = classifiers.ConvnetClassifier(len(keywords), 40).cuda()
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    print('Beginning Training')
    training_loss, validation_loss = train.train_classifier(
        model, loss_fn, training_set, validation_set, patience=2, lr=1e-4)
    print('Done Training')

    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(('Training Set', 'Validation Set'))
    plt.savefig('error.png')
    plt.show()
    print('Acheived {:%} accuracy on the training set.'
          .format(train.get_accuracy(model, training_set)))
    print('Acheived {:%} accuracy on the validation set.'
          .format(train.get_accuracy(model, validation_set)))
    print('Acheived {:%} accuracy on the testing set.'
          .format(train.get_accuracy(model, testing_set)))
    torch.save(model.state_dict(), 'model.pkl')


def test_expanded():
    fake, real = data_processing.load_data()
    keywords = data_processing.load_keywords()
    model = classifiers.ConvnetClassifier(len(keywords), 40)
    model.load_state_dict(torch.load('model.pkl'))
    fake_encoded = data_processing.encode_data(keywords, fake, 40)
    fake_data = data_processing.Headlines((fake_encoded, np.ones(len(fake))))
    real_encoded = data_processing.encode_data(keywords, real, 40)
    real_data = data_processing.Headlines((real_encoded, np.zeros(len(real))))
    print('Data Loaded')

    train_data = data_processing.load_tsv('train.tsv')
    valid_data = data_processing.load_tsv('valid.tsv')
    test_data = data_processing.load_tsv('test.tsv')
    print('Data Loaded')

    train_encoded = data_processing.encode_data(keywords, train_data[0], 40)
    training_set = data_processing.Headlines((train_encoded, train_data[1]))
    print('Training Set Processed')
    valid_encoded = data_processing.encode_data(keywords, valid_data[0], 40)
    validation_set = data_processing.Headlines((valid_encoded, valid_data[1]))
    print('Validation Set Processed')
    test_encoded = data_processing.encode_data(keywords, test_data[0], 40)
    testing_set = data_processing.Headlines((test_encoded, test_data[1]))
    print('Testing Set Processed')

    print('Acheived {:%} accuracy on the fake set.'
          .format(train.get_accuracy(model, fake_data)))
    print('Acheived {:%} accuracy on the real set.'
          .format(train.get_accuracy(model, real_data)))
    print('Acheived {:%} accuracy on the training set.'
          .format(train.get_accuracy(model, training_set)))
    print('Acheived {:%} accuracy on the validation set.'
          .format(train.get_accuracy(model, validation_set)))
    print('Acheived {:%} accuracy on the testing set.'
          .format(train.get_accuracy(model, testing_set)))


def largest_activations():
    fake, real = data_processing.load_data()
    _, keywords = data_processing.process_data(fake, real)
    model = classifiers.ConvnetClassifier(len(keywords), 40)
    model.load_state_dict(torch.load('model_orig.pkl'))
    weights = model.classifier[1].weight.data.numpy()

    print("Real sequences")
    most_real = np.argsort(weights[0])[-10:]
    for most in most_real:
        if most < 100:
            conv = model.features3[0].weight.data.numpy()[most]
        elif most < 200:
            conv = model.features4[0].weight.data.numpy()[most - 100]
        else:
            conv = model.features5[0].weight.data.numpy()[most - 200]
        print(*keywords[np.argmax(conv, 0)])

    print("Fake sequences")
    most_fake = np.argsort(weights[1])[-10:]
    for most in most_fake:
        if most < 100:
            conv = model.features3[0].weight.data.numpy()[most]
        elif most < 200:
            conv = model.features4[0].weight.data.numpy()[most - 100]
        else:
            conv = model.features5[0].weight.data.numpy()[most - 200]
        print(*keywords[np.argmax(conv, 0)])

def main():
    np.random.seed(11)
    torch.manual_seed(11)
    train_original()


if __name__ == '__main__':
    main()
