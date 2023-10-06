import os
import numpy as np
import explainability_techniques.PDP as pdp
from sklearn.neighbors import KNeighborsClassifier


def vecPredictProba(models, X):
    probas = []
    for model in models:
        probas.append(model.predict_proba(X))
    probas = np.ravel(probas)[1::2]
    probas = np.column_stack(np.split(probas, len(models)))
    return probas


class CustomPlanner:
    def __init__(self, X, n_neighbors,
                 reqClassifiers, targetConfidence,
                 controllableFeaturesNames,
                 controllableFeatureIndices,
                 controllableFeatureDomains,
                 optimizationDirections,
                 scoreFunction, delta=1, plotsPath=None):

        self.n_neighbors = n_neighbors
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.controllableFeatureIndices = np.array(controllableFeatureIndices)
        self.controllableFeatureDomains = controllableFeatureDomains
        self.optimizationDirections = optimizationDirections
        self.scoreFunction = scoreFunction
        self.delta = delta

        # train a k nearest neighbors classifier only used to find the neighbors of a sample in the dataset
        knn = KNeighborsClassifier()
        knn.fit(X, np.zeros((X.shape[0], 1)))
        self.knn = knn

        # make pdps
        pdps = {}
        for i, feature in enumerate(controllableFeaturesNames):
            pdps[i] = []
            for j, reqClassifier in enumerate(reqClassifiers):
                path = None
                if plotsPath is not None:
                    path = plotsPath + "/req_" + str(j)
                    if not os.path.exists(path):
                        os.makedirs(path)
                pdps[i].append(pdp.partialDependencePlot(reqClassifier, X, [feature], "both", path + "/" + feature + ".png"))

        # make summary pdps
        self.summaryPdps = []
        for i, feature in enumerate(controllableFeaturesNames):
            path = None
            if plotsPath is not None:
                path = plotsPath + "/summary"
                if not os.path.exists(path):
                    os.makedirs(path)
            self.summaryPdps.append(pdp.multiplyPdps(pdps[i], path + "/" + feature + ".png"))

    def optimizeScoreStep(self, adaptation, confidence, excludedFeatures):
        newAdaptation = np.copy(adaptation)

        # select a feature to modify
        featureIndex = None
        domainIndex = None
        minConfidenceLoss = None
        for i, index in enumerate(self.controllableFeatureIndices):
            if index not in excludedFeatures:
                neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]
                slope = pdp.getSlope(self.summaryPdps[i], adaptation[index], neighborIndex)
                confidenceLoss = slope * self.optimizationDirections[i]
                if minConfidenceLoss is None or confidenceLoss < minConfidenceLoss:
                    featureIndex = index
                    domainIndex = i
                    minConfidenceLoss = confidenceLoss

        # return if no feature can be improved
        if featureIndex is None:
            return None, None

        # modify the selected feature
        newAdaptation[featureIndex] += self.optimizationDirections[featureIndex] * self.delta

        featureMin = self.controllableFeatureDomains[domainIndex, 0]
        featureMax = self.controllableFeatureDomains[domainIndex, 1]

        if newAdaptation[featureIndex] < featureMin:
            newAdaptation[featureIndex] = featureMin
            excludedFeatures.append(featureIndex)
        elif newAdaptation[featureIndex] > featureMax:
            newAdaptation[featureIndex] = featureMax
            excludedFeatures.append(featureIndex)

        newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])

        if (newConfidence < self.targetConfidence).any():
            newAdaptation = np.copy(adaptation)
            newConfidence = np.copy(confidence)
            excludedFeatures.append(featureIndex)

        return newAdaptation, newConfidence

    def optimizeConfidenceStep(self, adaptation, confidence, excludedFeatures):
        # search a better adaptation in the neighbourhood of the old one
        bestAdaptation = None
        bestConfidence = None
        maxConfidenceGain = None
        for i, index in enumerate(self.controllableFeatureIndices):
            if index not in excludedFeatures:
                newAdaptation = np.copy(adaptation)
                newAdaptation[index] -= self.optimizationDirections[index] * self.delta

                featureMin = self.controllableFeatureDomains[i, 0]
                featureMax = self.controllableFeatureDomains[i, 1]

                if newAdaptation[index] < featureMin:
                    newAdaptation[index] = featureMin
                    excludedFeatures.append(index)
                elif newAdaptation[index] > featureMax:
                    newAdaptation[index] = featureMax
                    excludedFeatures.append(index)

                newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])
                confidenceGain = np.sum(newConfidence - confidence)

                if maxConfidenceGain is None or confidenceGain > maxConfidenceGain:
                    maxConfidenceGain = confidenceGain
                    bestAdaptation = np.copy(newAdaptation)
                    bestConfidence = np.copy(newConfidence)

        # return if there is no trivial better adaptation
        if maxConfidenceGain is None:
            return None, confidence

        print("Confidence gain: " + str(maxConfidenceGain))
        return bestAdaptation, bestConfidence

    def findAdaptation(self, row):
        n_controllableFeatures = len(self.controllableFeatureIndices)

        # find neighbors
        print(self.knn.kneighbors([row], self.n_neighbors))
        neighbors = np.ravel(self.knn.kneighbors([row], self.n_neighbors, False))

        # starting solutions
        adaptations = np.empty((self.n_neighbors, len(row)))
        for i in range(self.n_neighbors):
            neighborIndex = neighbors[i]
            adaptation = np.copy(row)

            for j, index in enumerate(self.controllableFeatureIndices):
                adaptation[index] = pdp.getMaxPointOfLine(self.summaryPdps[j], neighborIndex)
                neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]

            adaptations[i] = adaptation

        adaptationsConfidence = vecPredictProba(self.reqClassifiers, adaptations)

        print(adaptations[:, :n_controllableFeatures])
        print(adaptationsConfidence)
        print([self.scoreFunction(a) for a in adaptations])

        validAdaptationIndices = []
        for i, confidence in enumerate(adaptationsConfidence):
            if (confidence >= self.targetConfidence).all():
                validAdaptationIndices.append(i)

        validAdaptations = adaptations[validAdaptationIndices]
        validAdaptationsConfidence = adaptationsConfidence[validAdaptationIndices]
        validAdaptationFound = len(validAdaptations) > 0

        if validAdaptationFound:
            bestAdaptationIndex = 0  # TODO use TA (the ranking algorithm) to get best adaptation based on confidence and score
            adaptation = validAdaptations[bestAdaptationIndex]
            confidence = validAdaptationsConfidence[bestAdaptationIndex]
        else:
            bestAdaptationIndex = 0  # TODO use TA (the ranking algorithm) to get best adaptation based on confidence only
            adaptation = adaptations[bestAdaptationIndex]
            confidence = adaptationsConfidence[bestAdaptationIndex]

        print(adaptation)
        print(confidence)

        # enhance solution
        steps = 0
        excludedFeatures = []
        if validAdaptationFound:
            while len(excludedFeatures) < n_controllableFeatures:
                adaptation, confidence = self.optimizeScoreStep(adaptation, confidence, excludedFeatures)
                steps += 1
        else:
            while adaptation is not None and (confidence < self.targetConfidence).any():
                adaptation, confidence = self.optimizeConfidenceStep(adaptation, confidence, excludedFeatures)
                steps += 1

        print("Total steps: " + str(steps))

        print(confidence)
        """
        if (confidence < self.targetConfidence).any():
            adaptation = None
            confidence = None
        """

        return adaptation, confidence
