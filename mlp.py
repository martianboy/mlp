import numpy as np


def encode(cat, n):
    ret = [0] * n
    ret[cat] = 1
    return tuple(ret)


class MLP:
    def __init__(self, shape, activationFuncs, derivationFuncs):
        self.shape = np.array(shape)
        self.layersCount = len(shape) - 1
        self.activationFunctions = activationFuncs
        self.derivationFunctions = derivationFuncs

        self.eta = 0.1
        self.alpha = 0.01

        self.nodeValues = [np.ones((1, n)) for n in shape]
        self.deltas = [np.zeros((1, n)) for n in shape[1:]]
        self.weights = []
        self.deltaWeights = []

        np.random.seed()

        for k in range(self.layersCount):
            Wbar = (np.random.rand(shape[k] + 1, shape[k + 1]) - 0.5) * 10
            self.weights.append(Wbar)
            self.deltaWeights.append(np.zeros((shape[k] + 1, shape[k + 1])))

    def resetWeights(self):
        np.random.seed()
        for k in range(self.layersCount):
            Wbar = (np.random.rand(self.shape[k] + 1, self.shape[k + 1]) - 0.5) * 2
            self.weights[k][:, :] = Wbar

    def train(self, dataset, epsilon=35, epochs=2500, eta=0.25, alpha=0.4):
        self.eta = eta
        self.alpha = alpha

        epoch = 0
        stopN = 0

        while True:                 # Epochs
            epoch += 1
            epochError = 0

            stopTol = 0

            np.random.shuffle(dataset)

            for pattern, desired in dataset:
                out = self.feedForward(pattern)
                mappedOut = np.array(encode(out.argmax(), self.shape[-1]))

                #print('desired: ', desired, ' - mappedOut: ', mappedOut, ' - out: ', out)

                if (mappedOut != desired).any():

                    # epochError += 0.5 * ||out - desired||**2
                    error = desired - out
                    epochError += 1

#                    if stopN > 0:
#                        stopTol += 1

#                    if stopTol > 5:
#                        stopTol = 0
#                        stopN = 0

                    self.backpropagate(error)

                    self.updateWeights()

            #print(epoch, ': ', epochError, ' stopN: ', stopN)

            if epochError <= epsilon:
                stopN += 1
            else:
                stopN = 0

            if stopN == 10:
                print('Converged at epoch ', epoch)
                break

            if epoch >= epochs:
                print('Did not converge after ', epoch, ' epochs.')
                break

            #self.eta = (np.exp(-(maxEpError - epochError / len(dataset)) * np.exp(1) / maxEpError))
            s = float(epochError) / len(dataset)
            self.eta = 1.5 * np.exp(-(0.05 * s + 2.6 * (1 - s)))
            if np.random.randint(25) == 5:
                print(epoch, ' - Epoch Error: ', epochError, ', Eta: ', self.eta, ', stopN: ', stopN)

        #print('Max Epoch Error: ', maxEpError)

    # returns the netowk output
    # saves all mid-layer output values in self.nodeValues
    def feedForward(self, pattern):
        o = self.nodeValues[0][:, :] = pattern

        for k in range(self.layersCount):
            W = self.weights[k]
            ohat = np.concatenate((o, (1,)))

            o = ohat.dot(W)
            if self.activationFunctions[k] != None:
                o = self.activationFunctions[k](o)
            self.nodeValues[k + 1][:, :] = o

        return o

    def backpropagate(self, error):
        self.deltas[-1][:, :] = delta = np.atleast_2d(error)

        dsigmoid = self.derivationFunctions[-1]
        if dsigmoid != None:
            self.deltas[-1][:, :] = delta = delta * dsigmoid(self.nodeValues[-1])

        for k in range(self.layersCount - 1, 0, -1):
            delta = np.dot(delta, self.weights[k][:-1].T)

            dsigmoid = self.derivationFunctions[k - 1]
            if dsigmoid != None:
                delta = delta * dsigmoid(self.nodeValues[k])

            self.deltas[k - 1][:, :] = delta

    def updateWeights(self):
        for k in range(self.layersCount):
            ohat = np.ones((1, self.shape[k] + 1))
            ohat[:, :-1] = self.nodeValues[k]
            delta = np.atleast_2d(self.deltas[k])

            #dW = np.dot(ohat.T, delta)
            dW = self.eta * np.dot(ohat.T, delta) + self.alpha * self.deltaWeights[k]
            self.weights[k] += dW
            #self.eta * dW + self.alpha * self.deltaWeights[k]
            self.deltaWeights[k][:, :] = dW
