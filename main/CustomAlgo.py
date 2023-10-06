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

        # return if no feature can be modified
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

        newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])[0]

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

                newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])[0]

                confidenceGains = newConfidence - confidence
                for j, c in enumerate(newConfidence):
                    if confidenceGains[j] < 0 and c >= self.targetConfidence[j]:
                        confidenceGains[j] = 0
                confidenceGain = np.sum(confidenceGains)

                if maxConfidenceGain is None or confidenceGain > maxConfidenceGain:
                    maxConfidenceGain = confidenceGain
                    bestAdaptation = np.copy(newAdaptation)
                    bestConfidence = np.copy(newConfidence)

        # return if no feature can be modified
        if maxConfidenceGain is None:
            return None, confidence

        # print("Confidence gain: " + str(maxConfidenceGain))
        return bestAdaptation, bestConfidence

    def findAdaptation(self, row):
        n_controllableFeatures = len(self.controllableFeatureIndices)

        # find neighbors
        # print(self.knn.kneighbors([row], self.n_neighbors))
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

        print("\nStarting adaptations:")
        print(adaptations[:, :n_controllableFeatures])
        print("\nStarting adaptations confidence:")
        print(adaptationsConfidence)
        print("\nStarting adaptations score:")
        print([self.scoreFunction(a) for a in adaptations])

        validAdaptationIndices = []
        for i, confidence in enumerate(adaptationsConfidence):
            if (confidence >= self.targetConfidence).all():
                validAdaptationIndices.append(i)

        validAdaptations = adaptations[validAdaptationIndices]
        validAdaptationsConfidence = adaptationsConfidence[validAdaptationIndices]
        validAdaptationFound = len(validAdaptations) > 0

        if validAdaptationFound:
            def startingSolutionScore(a, c):
                return self.scoreFunction(a) + np.sum(c - self.targetConfidence)

            validAdaptationsScores = [startingSolutionScore(validAdaptations[i], validAdaptationsConfidence[i])
                                      for i in range(len(validAdaptations))]
            bestAdaptationIndex = np.where(np.max(validAdaptationsScores))[0][0]

            print("\nStarting valid adaptations ranking")
            print(validAdaptationsScores)
            print("Best starting valid adaptation: " + str(bestAdaptationIndex))

            adaptation = validAdaptations[bestAdaptationIndex]
            confidence = validAdaptationsConfidence[bestAdaptationIndex]
        else:
            def startingSolutionScore(c):
                return np.sum(c - self.targetConfidence)

            adaptationsScores = [startingSolutionScore(adaptationsConfidence[i])
                                 for i in range(len(adaptations))]
            bestAdaptationIndex = np.where(np.max(adaptationsScores))[0][0]

            print("\nStarting adaptations ranking")
            print(adaptationsScores)
            print("Best starting adaptation: " + str(bestAdaptationIndex))

            adaptation = adaptations[bestAdaptationIndex]
            confidence = adaptationsConfidence[bestAdaptationIndex]

        print(adaptation)
        print("With confidence:")
        print(confidence)

        # enhance solution
        confidenceSteps = 0
        scoreSteps = 0

        # optimize confidence the adaptation is not valid
        excludedFeatures = []
        if not validAdaptationFound:
            while adaptation is not None and (confidence < self.targetConfidence).any():
                adaptation, confidence = self.optimizeConfidenceStep(adaptation, confidence, excludedFeatures)
                # print(confidence)
                confidenceSteps += 1

        # then optimize score if the adaptation is valid
        excludedFeatures = []
        if adaptation is not None:
            while len(excludedFeatures) < n_controllableFeatures:
                adaptation, confidence = self.optimizeScoreStep(adaptation, confidence, excludedFeatures)
                scoreSteps += 1

        print("\nConfidence optimization steps: " + str(confidenceSteps))
        print("Score optimization steps:      " + str(scoreSteps))
        print("Total optimization steps:      " + str(confidenceSteps + scoreSteps))

        print("\nFinal confidence:")
        print(confidence)
        print()

        return adaptation, confidence
