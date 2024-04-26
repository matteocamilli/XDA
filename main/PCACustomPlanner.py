import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from CustomPlanner import CustomPlanner
from util import vecPredictProba, cartesian_product
from explainability_techniques.PCA import pcaClassifier
import explainability_techniques.PDP as pdp


class PCACustomPlanner(CustomPlanner):

    def __init__(self, X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                 controllableFeaturesNames,
                 controllableFeatureIndices, controllableFeatureDomains, optimizationDirections,
                 optimizationScoreFunction, delta=1, plotsPath=None):
        super().__init__(X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                         controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                         optimizationDirections, optimizationScoreFunction, delta, plotsPath)


        startTime = time.time()
        scores = pcaClassifier(X, len(controllableFeatureIndices))
        self.controllableFeatureIndices = np.argsort(scores)[::-1]
        print(scores)
        print(self.controllableFeatureIndices)
        endTime = time.time()
        print("PCA classifier duration:             " + str(endTime - startTime) + " s")
        print("=" * 100)

    def optimizeScoreStep(self, adaptation, confidence, isValidAdaptation, neighborIndex, excludedFeatures,
                          tempExcludedFeatures):

        featureIndex = None
        for i in self.controllableFeatureIndices:
            if i not in excludedFeatures and i not in tempExcludedFeatures:
                featureIndex = i

        if featureIndex is None:
            return None, None

        # modify the selected feature
        newAdaptation = np.copy(adaptation)
        newAdaptation[featureIndex] += self.optimizationDirections[featureIndex] * self.delta

        featureMin = self.controllableFeatureDomains[featureIndex, 0]
        featureMax = self.controllableFeatureDomains[featureIndex, 1]

        if newAdaptation[featureIndex] < featureMin:
            newAdaptation[featureIndex] = featureMin
            excludedFeatures.append(featureIndex)
        elif newAdaptation[featureIndex] > featureMax:
            newAdaptation[featureIndex] = featureMax
            excludedFeatures.append(featureIndex)

        newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])[0]

        if (isValidAdaptation and (newConfidence < self.targetConfidence).any()) \
                or (not isValidAdaptation and (newConfidence < confidence).any()):
            newAdaptation = np.copy(adaptation)
            newConfidence = np.copy(confidence)
            tempExcludedFeatures.append(featureIndex)
        else:
            tempExcludedFeatures.clear()

        return newAdaptation, newConfidence
