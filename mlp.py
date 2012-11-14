import numpy as np

class NNet:
    def __init__(self, structure):
        self.structure = structure
        self.weights = []

        np.random.seed()

        for k in range(1,len(structure)):
            self.weights.append(np.random.rand(self.structure[k-1],self.structure[k]) - 0.5

    def train(self, dataset, epsilon = 0.00005, N = 50):
        while True:
            epochError = 0
            for pattern,desiredValues in dataset:
                out = self.feedForward(pattern)
                deltaOut = map(lambda o,d: o*(1-o) * (d-o), out, desiredValues)

                epochError += sum(map(lambda o,d: (o-d)**2, out, desiredValues)) / 2
                
                deltaHidden = self.backpropagate(deltaOut)
                self.updateWeights(deltaHidden)

            if epochError < epsilon: break

    def feedForward(self, pattern):
        X = pattern
        for weightsOfLayer in self.weights
            X = weightsOfLayer.dot(X)
        return X

    def backpropagate(self, deltaOut):
        pass
    def updateWeights(self, deltaHidden):
        pass
