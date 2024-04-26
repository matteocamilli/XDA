from explainability_techniques.FeatureImportance import permutation_importance_classifier
import time
import numpy as np
from CustomPlanner import CustomPlanner
from sklearn.neighbors import KNeighborsClassifier
import explainability_techniques.PDP as pdp
from util import vecPredictProba, cartesian_product


class FICustomPlanner(CustomPlanner):

    def __init__(self, X, Y, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                 controllableFeaturesNames,
                 controllableFeatureIndices, controllableFeatureDomains, optimizationDirections,
                 optimizationScoreFunction, delta=1, plotsPath=None):
        super().__init__(X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                         controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                         optimizationDirections, optimizationScoreFunction, delta, plotsPath)

        startTime = time.time()

        cumulative_importance = np.zeros(len(controllableFeatureIndices))

        for i, reqClassifier in enumerate(self.reqClassifiers):

            feature_indices = permutation_importance_classifier(reqClassifier, X, Y, controllableFeatureIndices)

            for j in range(len(feature_indices)):
                cumulative_importance[j] += feature_indices[j]

        self.controllableFeatureIndices = np.argsort(cumulative_importance)[::-1]
        print(cumulative_importance)
        print(self.controllableFeatureIndices)
        endTime = time.time()
        print("FI classifier duration: " + str(endTime - startTime) + " s")
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



