import os
import numpy as np
import explainability_techniques.PDP as pdp
from sklearn.neighbors import KNeighborsClassifier
from util import vecPredictProba
from util import cartesian_product


class CustomPlanner:
    def __init__(self, X, n_neighbors,
                 reqClassifiers, targetConfidence,
                 controllableFeaturesNames,
                 controllableFeatureIndices,
                 controllableFeatureDomains,
                 optimizationDirections,
                 successScoreFunction,
                 optimizationScoreFunction,
                 delta=1, plotsPath=None):

        self.n_neighbors = n_neighbors
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.controllableFeatureIndices = np.array(controllableFeatureIndices)
        self.controllableFeatureDomains = controllableFeatureDomains
        self.optimizationDirections = optimizationDirections
        self.successScoreFunction = successScoreFunction
        self.optimizationScoreFunction = optimizationScoreFunction
        self.delta = delta

        # train a k nearest neighbors classifier only used to find the neighbors of a sample in the dataset
        knn = KNeighborsClassifier()
        knn.fit(X, np.zeros((X.shape[0], 1)))
        self.knn = knn

        # make pdps
        self.pdps = {}
        for i, feature in enumerate(controllableFeaturesNames):
            self.pdps[i] = []
            for j, reqClassifier in enumerate(reqClassifiers):
                path = None
                if plotsPath is not None:
                    path = plotsPath + "/req_" + str(j)
                    if not os.path.exists(path):
                        os.makedirs(path)
                self.pdps[i].append(pdp.partialDependencePlot(reqClassifier, X, [feature], "both", path + "/" + feature + ".png"))

        # make summary pdps
        self.summaryPdps = []
        for i, feature in enumerate(controllableFeaturesNames):
            path = None
            if plotsPath is not None:
                path = plotsPath + "/summary"
                if not os.path.exists(path):
                    os.makedirs(path)
            self.summaryPdps.append(pdp.multiplyPdps(self.pdps[i], path + "/" + feature + ".png"))

    def optimizeScoreStep(self, adaptation, confidence, excludedFeatures):
        # select a feature to modify
        featureIndex = None
        controllableIndex = None
        minConfidenceLoss = None
        for i, index in enumerate(self.controllableFeatureIndices):
            if index not in excludedFeatures:
                neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]
                slope = pdp.getSlope(self.summaryPdps[i], adaptation[index], neighborIndex)
                confidenceLoss = slope * self.optimizationDirections[i]
                if minConfidenceLoss is None or confidenceLoss < minConfidenceLoss:
                    featureIndex = index
                    controllableIndex = i
                    minConfidenceLoss = confidenceLoss

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

    def findAdaptation(self, row):
        n_controllableFeatures = len(self.controllableFeatureIndices)
        n_startingSolutions = 10

        # find neighbors
        neighbors = np.ravel(self.knn.kneighbors([row], self.n_neighbors, False))

        # starting solutions
        adaptations = [row]
        for i in range(self.n_neighbors):
            neighborIndex = neighbors[i]
            adaptation = np.copy(row)

            recalculateNeighbor = False
            excludedFeatures = []
            while len(excludedFeatures) < n_controllableFeatures:
                # recalculate neighbor after the first step
                if recalculateNeighbor:
                    neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]
                recalculateNeighbor = True

                maxYVals = [(j, pdp.getMaxOfLine(self.summaryPdps[j], neighborIndex)) for j in range(n_controllableFeatures)]
                maxYVals = sorted(maxYVals, key=lambda val: val[1], reverse=True)

                for val in maxYVals:
                    controllableIndex = val[0]
                    if controllableIndex not in excludedFeatures:
                        maximals = pdp.getMaximalsOfLine(self.summaryPdps[controllableIndex], neighborIndex)
                        if self.optimizationDirections[controllableIndex] == -1:
                            # leftmost maximal
                            x = maximals[0]
                        else:
                            # rightmost maximal
                            x = maximals[len(maximals) - 1]

                        newAdaptation = np.copy(adaptation)
                        newAdaptation[self.controllableFeatureIndices[controllableIndex]] = x
                        adaptations.append(newAdaptation)
                        excludedFeatures.append(controllableIndex)
                        break

        # remove duplicate solutions
        adaptations = np.unique(adaptations, axis=0)

        for adaptation in adaptations:
            neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]

            maximals = [pdp.getMaximalsOfLine(self.summaryPdps[i], neighborIndex) for i in
                        range(n_controllableFeatures)]

            maxPossibilities = 10000
            n_possibilities = np.prod([len(m) for m in maximals])
            while n_possibilities > maxPossibilities:
                i = np.argmax([len(m) for m in maximals])
                maximals[i] = maximals[i][1::2]
                n_possibilities = np.prod([len(m) for m in maximals])

            possibilities = cartesian_product(*maximals)
            possibilities = np.append(possibilities,
                                      np.repeat([row[n_controllableFeatures:]], possibilities.shape[0], axis=0),
                                      axis=1)

            """
            print(len(possibilities))
            p = vecPredictProba(self.reqClassifiers, possibilities, axis=1)
            print(np.max(p))
            """

            adaptations = np.append(adaptations, possibilities, axis=0)

        # remove duplicate solutions again
        adaptations = np.unique(adaptations, axis=0)

        adaptationsConfidence = vecPredictProba(self.reqClassifiers, adaptations)

        """
        print("\nStarting adaptations:")
        print(adaptations[:, :n_controllableFeatures])
        print("\nStarting adaptations confidence:")
        print(adaptationsConfidence)
        print("\nStarting adaptations score:")
        print([self.scoreFunction(a) for a in adaptations])
        """

        validAdaptationIndices = []
        for i, confidence in enumerate(adaptationsConfidence):
            if (confidence >= self.targetConfidence).all():
                validAdaptationIndices.append(i)

        validAdaptations = adaptations[validAdaptationIndices]
        validAdaptationsConfidence = adaptationsConfidence[validAdaptationIndices]
        validAdaptationFound = len(validAdaptations) > 0

        if validAdaptationFound:
            def solutionScore(a, c):
                return self.optimizationScoreFunction(a) + np.sum(c - self.targetConfidence)

            validAdaptationsScores = [solutionScore(validAdaptations[i], validAdaptationsConfidence[i])
                                      for i in range(len(validAdaptations))]
            bestAdaptationIndices = np.argpartition(validAdaptationsScores, -n_startingSolutions)[-n_startingSolutions:]

            """
            print("\nStarting valid adaptations ranking")
            print(validAdaptationsScores)
            """
            print("Best starting valid adaptations: " + str(bestAdaptationIndices))

            bestAdaptations = validAdaptations[bestAdaptationIndices]
            bestAdaptationsConfidence = validAdaptationsConfidence[bestAdaptationIndices]
        else:
            def solutionScore(c):
                return np.sum(c - self.targetConfidence)

            adaptationsScores = [solutionScore(adaptationsConfidence[i])
                                 for i in range(len(adaptations))]
            bestAdaptationIndices = np.argpartition(adaptationsScores, -n_startingSolutions)[-n_startingSolutions:]

            """
            print("\nStarting adaptations ranking")
            print(adaptationsScores)
            """
            print("Best starting adaptations: " + str(bestAdaptationIndices))

            bestAdaptations = adaptations[bestAdaptationIndices]
            bestAdaptationsConfidence = adaptationsConfidence[bestAdaptationIndices]

        print(bestAdaptations[:, :n_controllableFeatures])
        print("With confidence:")
        print(bestAdaptationsConfidence)

        # enhance solutions
        optimizationSteps = [0] * len(bestAdaptations)

        # optimize score if the adaptations are valid
        if validAdaptationFound:
            for i, adaptation, confidence in enumerate(bestAdaptations, bestAdaptationsConfidence):
                excludedFeatures = []
                while len(excludedFeatures) < n_controllableFeatures:
                    adaptation, confidence = self.optimizeScoreStep(adaptation, confidence, excludedFeatures)
                    optimizationSteps[i] += 1

                bestAdaptations[i] = adaptation
                bestAdaptationsConfidence[i] = confidence

        print("\nScore optimization steps: " + str(optimizationSteps))

        print("Final adaptations:")
        print(bestAdaptations[:, :n_controllableFeatures])

        print("\nFinal confidence:")
        print(bestAdaptationsConfidence)

        if validAdaptationFound:
            bestAdaptationsScores = [solutionScore(bestAdaptations[i], bestAdaptationsConfidence[i])
                                     for i in range(len(bestAdaptations))]
        else:
            bestAdaptationsScores = [solutionScore(bestAdaptations[i]) for i in range(len(bestAdaptations))]

        finalAdaptationIndex = np.argmax(bestAdaptationsScores)
        finalAdaptation = bestAdaptations[finalAdaptationIndex]
        finalAdaptationConfidence = bestAdaptationsConfidence[finalAdaptationIndex]
        print("\nBest final adaptation:")
        print(finalAdaptation[:n_controllableFeatures])
        print("With confidence:")
        print(finalAdaptationConfidence)

        return finalAdaptation, finalAdaptationConfidence, self.optimizationScoreFunction(finalAdaptation)
