#!/usr/bin/python3

import sys

import numpy as np
import torch
from torch.autograd import Variable

import classifiers
import data_processing

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


def classify_headlines(model, encoded):
    x = torch.unsqueeze(encoded, 0)
    y = model(x).data.numpy()[0]
    return np.argmax(y)


def encode_headline(keywords, headline, dim):
    encoded = np.zeros((len(keywords), dim))
    for i, word in enumerate(headline):
        j = np.where(keywords == word)[0]
        if j.size > 0:
            encoded[j[0], i] = 1
    return Variable(torch.from_numpy(encoded).type(dtype_float))


if __name__ == '__main__':
    keywords = data_processing.load_keywords()
    model = classifiers.ConvnetClassifier(len(keywords), 40)
    model.load_state_dict(torch.load('model_orig.pkl'))
    with open(sys.argv[1]) as f:
        for headline in f:
            encoded = encode_headline(keywords, headline.split(' '), 40)
            print(classify_headlines(model, encoded))
