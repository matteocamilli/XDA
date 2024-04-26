import time

import numpy as np
from CustomPlanner import CustomPlanner
from util import vecPredictProba
from explainability_techniques.SHAP import shapClassifier
import explainability_techniques.PDP as pdp


class SHAPCustomPlanner(CustomPlanner):

    def __init__(self, X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence, controllableFeaturesNames,
                 controllableFeatureIndices, controllableFeatureDomains, optimizationDirections,
                 optimizationScoreFunction, delta=1, plotsPath=None):

        super().__init__(X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                         controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                         optimizationDirections, optimizationScoreFunction, delta, plotsPath)

        startTime = time.time()
        """
        cumulative_importance = np.zeros(len(controllableFeatureIndices))

        for i, reqClassifier in enumerate(self.reqClassifiers):
            feature_indices = shapClassifier(reqClassifier, X, controllableFeatureIndices)
            for j in range(len(feature_indices)):
                cumulative_importance[j] += feature_indices[j]

        self.controllableFeatureIndices = np.argsort(cumulative_importance)[::-1]
        print(cumulative_importance)
        print(self.controllableFeatureIndices)
        """
        self.controllableFeatureIndices = [2, 1, 0]
        endTime = time.time()
        print("SHAP classifier duration: " + str(endTime - startTime) + " s")
        print("=" * 100)

    def optimizeScoreStep(self, adaptation, confidence, isValidAdaptation, neighborIndex, excludedFeatures,
                          tempExcludedFeatures):

        featureIndex = None
        for i in self.controllableFeatureIndices:
            if i not in excludedFeatures and i not in tempExcludedFeatures:
                featureIndex = i


            # return if no feature can be modified
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
