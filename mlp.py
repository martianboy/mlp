import numpy as np
from math import tanh

class MLP:
    def __init__(self, structure):
        self.structure = np.array(structure,np.uint8)
        self.weightMatrices = []

        # Node storage format: (excitation, derivative, delta)
        self.nodeStorage = [np.ndarray((nou,3)) for nou in self.structure[1:]]
        self.layersCount = len(structure)

        #self.alpha = 

        np.random.seed()

        for k in range(1,self.layersCount):
            self.weightMatrices.append(np.random.rand(self.structure[k-1],
                                                    self.structure[k]) - 0.5)

    # TODO: Take learning rate into account
    def train(self, dataset, epsilon = 0.00005, N = 50):
        while True:
            epochError = 0
            for pattern,desiredValues in dataset:
                
                # Feed the pattern forward in the network
                self.feedForward(pattern)

                out,outDeriv = self.nodeStorage[self.layersCount - 1]

                # Add quadratic deviation for this pattern to the epoch error
                epochError += (desiredValues - out)**2 / 2

                # Backpropagate error to the output layer units
                nodeStorage[self.layersCount - 2][:,3] =
                                            (desiredValues - out) * outDeriv
                # Backpropagate error to the hidden units
                self.backpropagate(deltaOutputLayer)

                # Update weights based on the computed deltas and learning rate
                self.updateWeights(deltaHidden)

            if epochError < epsilon: break

    def feedForward(self, pattern):
        sigmoid = np.vectorize(tanh)
        o = pattern
        for layer,weightsOfLayer in enumerate(self.weightMatrices):
            o = sigmoid(o.dot(weightsOfLayer))
            op = o*(1-o)
            self.nodeStorage[layer][:,1:2] = zip(o,op)

    def backpropagate(self, deltaOut):
        pass
    def updateWeights(self, deltaHidden):
        pass
