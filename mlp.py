import numpy as np


class MLP:
    def __init__(self, shape):
        self.shape = np.array(shape)
        self.layersCount = len(shape) - 1

        self.eta = 0.1
        self.alpha = 0.1

        self.sigmoid = lambda x: 1 / (1 + np.exp(-5 * x))
        self.dsigmoid = lambda o: 5 * o * (1 - o)

        self.nodeValues = [np.ones((1, n)) for n in shape]
        self.deltas = [np.zeros((1, n)) for n in shape[1:]]
        self.weights = []
        self.deltaWeights = []

        np.random.seed()

        for k in range(self.layersCount):
            Wbar = (np.random.rand(shape[k] + 1, shape[k + 1]) - 0.5) / 2
            self.weights.append(Wbar)
            self.deltaWeights.append(np.zeros((shape[k] + 1, shape[k + 1])))

    def train(self, dataset, epsilon=0.01, epochs=2500, eta=0.1, alpha=0.1):
        self.eta = eta
        self.alpha = alpha

        epoch = 0
        while True:         # Epochs
            epoch += 1
            epochError = 0

            np.random.shuffle(dataset)

            for pattern, desired in dataset:

                out = self.feedForward(pattern)

                # epochError += 0.5 * ||out - desired||**2
                error = out - desired
                epochError += 0.5 * np.linalg.norm(error) ** 2

                self.backpropagate(error)

                self.updateWeights()

            if epoch >= epochs:
                print('Did not converge after ', epoch, ' epochs.')
                break
            if epochError <= epsilon:
                print('Converged at epoch ', epoch)
                break

    # returns the netowk output
    # saves all mid-layer output values in self.nodeValues
    def feedForward(self, pattern):
        o = self.nodeValues[0] = pattern

        for k in range(self.layersCount):
            W = self.weights[k]
            ohat = np.concatenate((o, (1,)))

            o = self.sigmoid(ohat.dot(W))
            self.nodeValues[k + 1] = o

        return o

    def backpropagate(self, error):
        e = error
        for k in range(self.layersCount - 1, -1, -1):
            o = self.nodeValues[k + 1]
            D2 = np.diag(self.dsigmoid(o))

            self.deltas[k] = delta = D2.dot(e)

            W2 = self.weights[k][:-1]
            e = W2.dot(delta)

    def updateWeights(self):
        for k in range(self.layersCount):
            ohat = np.vstack(np.concatenate((self.nodeValues[k], (1,))))
            delta = np.atleast_2d(self.deltas[k])

            self.deltaWeights[k] = \
                -self.eta * ohat.dot(delta)  \
                + self.alpha * self.deltaWeights[k]

            self.weights[k] += self.deltaWeights[k]
