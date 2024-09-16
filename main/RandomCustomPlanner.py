import numpy as np
import random
from util import vecPredictProba


class RandomPlanner:

    def __init__(self, controllableIndices, controllableDomains, discreteIndices, models, optimizationScoreFunction):
        self.controllableIndices = controllableIndices
        self.controllableDomains = controllableDomains
        self.reqClassifiers = models
        self.optimizationScoreFunction = optimizationScoreFunction
        self.discreteIndices = discreteIndices

    def findAdaptation(self, row):
        bestAdaptation = row
        lb = self.controllableDomains[:, 0]
        ub = self.controllableDomains[:, 1]

        for i in self.controllableIndices:
            if i in self.discreteIndices:
                bestAdaptation[i] = random.randint(lb[i], ub[i])
            else:
                bestAdaptation[i] = np.random.uniform(lb[i], ub[i])

        confidence = vecPredictProba(self.reqClassifiers, [bestAdaptation])[0]

        score = self.optimizationScoreFunction(bestAdaptation)

        return bestAdaptation, confidence, score
