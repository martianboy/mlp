import numpy as np
from math import tanh

class MLP:
    def __init__(self, structure, alpha = 0.1):
        self.structure = np.array(structure,np.uint8)
        self.weightMatrices = []

        # Node storage format: (excitation, derivative, delta)
        self.nodeStorage = [np.ndarray((nou,3)) for nou in self.structure[1:]]
        self.layersCount = len(structure) - 1

        np.random.seed()

        for k in range(1,self.layersCount):
            self.weightMatrices.append(np.random.rand(self.structure[k-1] + 1,
                                                    self.structure[k]) - 0.5)

    # TODO: Take learning rate into account
    def train(self, dataset, epsilon = 0.00005, N = 50):
        while True:
            epochError = 0
            for pattern,desiredValues in dataset:
                
                # Feed the pattern forward in the network
                self.feedForward(pattern)

                out,outDeriv = \
                    np.dstack(self.nodeStorage[self.layersCount - 1][:,0:2])[0]

                # Add quadratic deviation for this pattern to the epoch error
                epochError += np.linalg.norm(desiredValues - out)**2 / 2

                # Backpropagate error to the output layer units
                self.nodeStorage[self.layersCount - 1][:,2] = \
                                            (desiredValues - out) * outDeriv
                # Backpropagate error to the hidden units
                self.backpropagate()

                # Update weights based on the computed deltas and learning rate
                self.updateWeights()

            if epochError < epsilon: break

    def feedForward(self, pattern):
        sigmoid = np.vectorize(tanh)
        o = pattern
        for layer,weightsOfLayer in enumerate(self.weightMatrices):
            o = np.concatenate((o,[1]))
            o = sigmoid(o.dot(weightsOfLayer))
            op = o*(1-o)
            self.nodeStorage[layer][:,0:2] = zip(o,op)

    def backpropagate(self):
        delta = self.nodeStorage[self.layersCount - 1][:,2]
        for layer in range(self.layersCount - 2, -1, -1):
            D = np.diag(self.nodeStorage[layer][:,2])
            W = self.weightMatrices[layer]
            self.nodeStorage[layer][:,2] = delta = D.dot(W).dot(delta)

    def updateWeights(self):
        for layer in range(self.layersCount - 1):
            delta = np.vstack(self.nodeStorage[layer][:,2])
            ohat = np.dstack(self.nodeStorage[layer][:,0])[0]
            self.weightMatrices[layer] += -self.alpha * delta.dot(ohat)
