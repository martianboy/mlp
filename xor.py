#import numpy as np
import mlp


def main():
    dataset = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

    m = mlp.MLP([2, 2, 1])
    m.train(dataset, eta=0.6, epochs=1000)

    print(m.feedForward((0, 0)))
    print(m.feedForward((0, 1)))
    print(m.feedForward((1, 0)))
    print(m.feedForward((1, 1)))

    for i in range(len(dataset)):
        o = m.feedForward(dataset[i][0])
        print(i, dataset[i][0], '%.2f' % o[0], ' (expected %.2f)' % dataset[i][1])

if __name__ == '__main__':
    main()
