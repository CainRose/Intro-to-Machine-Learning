import csv
import pickle
import string

import numpy as np
import torch
from scipy import sparse
from torch.autograd import Variable

dtype_float = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

label_dict = {
    'true': 0,
    'mostly-true': 0,
    'half-true': 0,
    'barely-true': 1,
    'false': 1,
    'pants-fire': 1
}


class Headlines():
    def __init__(self, data):
        self.data = data[0]
        self.label = data[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sentence = self.data[item].toarray()
        label = self.label[item]
        return sentence, label


class Loader():
    def __init__(self, dataset, step):
        self.dataset = dataset
        self.indices = np.random.permutation(len(dataset))
        self.step = step
        self.current = 0
        self.n_words, self.h_length = dataset[0][0].shape

    def __bool__(self):
        return self.current + self.step < len(self.dataset)

    def __iter__(self):
        self.current = 0
        self.indices = np.random.permutation(len(self.dataset))
        return self

    def __next__(self):
        if not self:
            raise StopIteration()
        data = np.empty((self.step, self.n_words, self.h_length))
        labels = np.empty(self.step)
        for i in range(self.step):
            d = self.dataset[self.indices[self.current + i]]
            data[i] = d[0]
            labels[i] = d[1]
        self.current += self.step
        return torch.from_numpy(data), torch.from_numpy(labels)


def load_data(fake_file='fake.txt', real_file='real.txt'):
    with open(fake_file) as f:
        fake = [s.split(' ') for s in f.read().splitlines()]
    with open(real_file) as f:
        real = [s.split(' ') for s in f.read().splitlines()]
    return fake, real


def load_tsv(file):
    headlines = []
    labels = []
    trans1 = str.maketrans('-', ' ')
    trans2 = str.maketrans('', '', string.punctuation)
    with open(file, encoding='utf8') as f:
        tsvin = csv.reader(f, delimiter='\t')
        for row in tsvin:
            labels.append(label_dict[row[1]])
            h = row[2].lower().translate(trans1).translate(trans2)
            headlines.append(h.split(' ')[:40])
    return headlines, labels


def load_keywords():
    with open('keywords.pkl', 'rb') as f:
        keywords = pickle.load(f)
    return keywords


def generate_keywords(*headlines, save=False):
    headlines = [headline for group in headlines for headline in group]
    keywords = sorted(list(set([w for h in headlines for w in h])))
    if save:
        with open('keywords.pkl', 'wb') as f:
            pickle.dump(keywords, f)
    return keywords


def encode_data(keywords, headlines, dim=None):
    data = np.empty(len(headlines), dtype=object)
    if not dim:
        dim = max([len(h) for h in headlines])
    for i, headline in enumerate(headlines):
        data[i] = np.zeros((len(keywords), dim))
        for j, w in enumerate(headline):
            if w in keywords:
                data[i][keywords.index(w), j] = 1
        data[i] = sparse.csr_matrix(data[i])
    return data


def process_data(fake, real):
    raw_data = fake + real
    labels = np.array([1] * len(fake) + [0] * len(real))
    keywords = sorted(list(
        set([w for headline in fake for w in headline]).union(
            set([w for headline in real for w in headline]))))
    data = np.empty(len(raw_data), dtype=object)
    l = max([len(h) for h in raw_data])
    for i, headline in enumerate(raw_data):
        data[i] = np.zeros((len(keywords), l))
        for j, w in enumerate(headline):
            data[i][keywords.index(w), j] = 1
        data[i] = sparse.csr_matrix(data[i])
    shuffled = np.random.permutation(len(labels))
    train_cutoff = np.floor(0.7 * len(labels)).astype(int)
    valid_cutoff = np.floor(0.85 * len(labels)).astype(int)
    train = (data[shuffled[:train_cutoff],], labels[shuffled[:train_cutoff],])
    valid = (data[shuffled[train_cutoff:valid_cutoff],],
             labels[shuffled[train_cutoff:valid_cutoff],])
    test = (data[shuffled[valid_cutoff:],], labels[shuffled[valid_cutoff:],])
    return (train, valid, test), np.array(keywords)
