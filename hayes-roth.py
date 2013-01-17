#!/usr/bin/env python
__author__ = 'Abbas Mashayekh <abbas.m@abbas-m.com>'

import signal
import sys

import csv
import numpy as np
import mlp


def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


def encode(cat, n):
    ret = [0] * n
    ret[cat - 1] = 1
    return tuple(ret)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    signal.signal(signal.SIGINT, signal_handler)

    dataset = []
    with open('./data/hayes-roth.data', 'rt') as datafile:
        dsreader = csv.reader(datafile)
        for r in dsreader:
            dataset.append((encode(int(r[2]), 4) + encode(int(r[3]), 4) + encode(int(r[4]), 4), encode(int(r[5]), 3)))

    sigm = np.vectorize(sigmoid)
    dsigm = np.vectorize(lambda o: o * (1 - o))
    #dtanh = np.vectorize(lambda o: 1 - o ** 2)

    m = mlp.MLP((3 * 4, 10, 3), (sigm, sigm), (dsigm, dsigm))
    m.train(dataset, epsilon=18, alpha=.4, eta=0.15, epochs=5000)

    with open('./data/hayes-roth.test', 'rt') as datafile:
        dsreader = csv.reader(datafile)
        modelTestErrors = 0
        for r in dsreader:
            pattern = (encode(int(r[1]), 4) + encode(int(r[2]), 4) + encode(int(r[3]), 4), encode(int(r[4]), 3))
            o = m.feedForward(pattern[0])
            o = np.array(encode(o.argmax() + 1, 3))
            e = ((o - pattern[1]) != 0)

            if e.any():
                modelTestErrors += 1
            print('Fed ', pattern[0], ' to the network, expected ', pattern[1], ', got ', o)

        print('Found ', modelTestErrors, ' classification errors.')

if __name__ == "__main__":
    main()
