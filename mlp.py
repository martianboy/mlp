import numpy as np


class MLP:
    def __init__(self, structure, learning_rate=0.1, momentum_effect=0.1):
        self.structure = np.array(structure, np.uint8)
        self.layersCount = len(structure) - 1

        self.learning_rate = learning_rate
        self.momentum_effect = momentum_effect

        self.epoch = 0
        self.pattern = []

        # Node storage format: (excitation, derivative, delta)
        self.storage = {'nodes': [], 'deltaWeights': []}
        self.storage['nodes'] = [np.zeros((nou, 3)) for nou in self.structure[1:]]

        np.random.seed()

        self.weightMatrices = []
        for k in range(self.layersCount):
            W = np.random.rand(structure[k] + 1, structure[k + 1])
            self.weightMatrices.append((W - 0.5) / 4)
            self.storage['deltaWeights'].append(np.zeros((structure[k] + 1, structure[k + 1])))

    def train(self, dataset, epsilon=0.0005, N=2500):
        while True:
            self.epoch += 1
            epochError = 0
            for pattern, desiredValues in dataset:

                # Feed the pattern forward in the network
                self.feedForward(pattern)

                out, outDeriv = np.dstack(self.storage['nodes'][-1][:, 0:2])[0]
                e = desiredValues - out

                # Add quadratic deviation for this pattern to the epoch error
                epochError += np.linalg.norm(e) ** 2 / 2

                # Backpropagate error to the output layer units
                self.storage['nodes'][-1][:, 2] = np.diag(outDeriv).dot(e)

                # Backpropagate error to the hidden units
                self.backpropagate()

                # Update weights based on the computed deltas and learning rate
                self.updateWeights()

            print(epochError)

            if epochError < epsilon:
                break
            if self.epoch >= N:
                break

    def feedForward(self, pattern):
        sigmoid = np.tanh
        self.pattern = pattern

        o = pattern
        for layer, weightsOfLayer in enumerate(self.weightMatrices):
            o = np.concatenate((o, [1]))
            o = sigmoid(o.dot(weightsOfLayer))
            op = 1 - o ** 2
            self.storage['nodes'][layer][:, 0:2] = list(zip(o, op))

    def backpropagate(self):
        delta = self.storage['nodes'][-1][:, 2]
        for layer in range(self.layersCount - 2, -1, -1):
            D = np.diag(self.storage['nodes'][layer][:, 1])
            W = self.weightMatrices[layer + 1][:-1]
            self.storage['nodes'][layer][:, 2] = delta = D.dot(W).dot(delta)

    def updateWeights(self):
        ohat = np.vstack(self.pattern + (1,))

        for layer in range(self.layersCount):
            delta = np.atleast_2d(self.storage['nodes'][layer][:, 2])
            self.storage['deltaWeights'][layer][:, :] = \
                self.momentum_effect * self.storage['deltaWeights'][layer] - \
                self.learning_rate * ohat.dot(delta)

            #print(self.weightMatrices[layer])

            self.weightMatrices[layer] += self.storage['deltaWeights'][layer]
            ohat = np.vstack(np.concatenate((self.storage['nodes'][layer][:, 0], (1,))))
