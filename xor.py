import numpy as np
#from os import chdir

#chdir('..')
from mlp import MLP


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hardlim(bias):
    return lambda x: float(x > bias)


def encode(cat, n):
    ret = [0] * n
    ret[cat] = 1
    return tuple(ret)


def main():
    dataset = [((0, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 0), (1, 0)), ((1, 1), (0, 1))]

    #dtanh = lambda o: 1 - o ** 2
    dsigm = lambda o: o * (1 - o)

    activation_functions = (np.vectorize(sigmoid), np.vectorize(sigmoid))
    #activation_functions = (np.tanh, np.tanh)
    derivation_functions = (np.vectorize(dsigm), np.vectorize(dsigm))
    #derivation_functions = (np.vectorize(dtanh), np.vectorize(dtanh))

    m = MLP((2, 3, 2), activation_functions, derivation_functions)
    m.train(dataset, epsilon=0, alpha=0.9, eta=.25, epochs=2500)

    for i in range(len(dataset)):
        o = m.feedForward(dataset[i][0])
        print(i, dataset[i][0], encode(o.argmax(), len(o)), ' (expected ', dataset[i][1], ')')

if __name__ == '__main__':
    main()
