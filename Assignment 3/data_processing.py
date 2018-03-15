import numpy as np
import torch
from torch.autograd import Variable
from scipy import sparse

dtype_float = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


def load_data():
    with open('clean_fake.txt') as f:
        fake = [s.split(' ') for s in f.read().splitlines()]
    with open('clean_real.txt') as f:
        real = [s.split(' ') for s in f.read().splitlines()]
    return fake, real


def process_data(fake, real, as_tensor=False):
    raw_data = fake + real
    labels = np.array([1] * len(fake) + [0] * len(real))
    keywords = sorted(list(set([w for headline in fake for w in headline])
                    .union(set([w for headline in real for w in headline]))))
    keywords = keywords[::-1]
    data = np.zeros((len(raw_data), len(keywords)))
    for i, headline in enumerate(raw_data):
        for w in headline:
            data[i, keywords.index(w)] = 1
    if as_tensor:
        data = Variable(torch.from_numpy(data).type(dtype_float),
                        requires_grad=False).cuda()
        labels = Variable(torch.from_numpy(labels).type(dtype_long),
                        requires_grad=False).cuda()
    else:
        data = sparse.csr_matrix(data)
    shuffled = np.random.permutation(len(labels))
    train_cutoff = np.floor(0.7 * len(labels)).astype(int)
    valid_cutoff = np.floor(0.85 * len(labels)).astype(int)
    train = (data[shuffled[:train_cutoff],], labels[shuffled[:train_cutoff],])
    valid = (data[shuffled[train_cutoff:valid_cutoff],],
             labels[shuffled[train_cutoff:valid_cutoff],])
    test = (data[shuffled[valid_cutoff:],], labels[shuffled[valid_cutoff:],])
    return (train, valid, test), np.array(keywords)
