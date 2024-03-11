import time

import numpy as np
from CustomPlanner import CustomPlanner
from util import vecPredictProba, cartesian_product
from explainability_techniques.SHAP import shapClassifier
import explainability_techniques.PDP as pdp


class SHAPCustomPlanner(CustomPlanner):

    def __init__(self, X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence, controllableFeaturesNames,
                 controllableFeatureIndices, controllableFeatureDomains, optimizationDirections,
                 optimizationScoreFunction, delta=1, plotsPath=None):
        super().__init__(X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                         controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                         optimizationDirections, optimizationScoreFunction, delta, plotsPath)

        self.sortedFeatures = {}
        startTime = time.time()
        for i, reqClassifier in enumerate(self.reqClassifiers):
            self.sortedFeatures[i] = shapClassifier(reqClassifier, X, controllableFeatureIndices)
        endTime = time.time()
        print("SHAP classifier duration:             " + str(endTime - startTime) + " s")

    def optimizeScoreStep(self, adaptation, confidence, isValidAdaptation, neighborIndex, excludedFeatures,
                          tempExcludedFeatures):

        # select a feature to modify
        featureIndex = None
        controllableIndex = None
        for i, index in enumerate(self.sortedFeatures):
            if i not in excludedFeatures and i not in tempExcludedFeatures:
                featureIndex = index
                controllableIndex = i

        # return if no feature can be modified
        if featureIndex is None:
            return None, None

        # modify the selected feature
        newAdaptation = np.copy(adaptation)
        newAdaptation[featureIndex] += self.optimizationDirections[controllableIndex] * self.delta

        featureMin = self.controllableFeatureDomains[controllableIndex, 0]
        featureMax = self.controllableFeatureDomains[controllableIndex, 1]

        if newAdaptation[featureIndex] < featureMin:
            newAdaptation[featureIndex] = featureMin
            excludedFeatures.append(controllableIndex)
        elif newAdaptation[featureIndex] > featureMax:
            newAdaptation[featureIndex] = featureMax
            excludedFeatures.append(controllableIndex)

        newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])[0]

        if (isValidAdaptation and (newConfidence < self.targetConfidence).any()) \
                or (not isValidAdaptation and (newConfidence < confidence).any()):
            newAdaptation = np.copy(adaptation)
            newConfidence = np.copy(confidence)
            tempExcludedFeatures.append(controllableIndex)
        else:
            tempExcludedFeatures.clear()

        return newAdaptation, newConfidence

