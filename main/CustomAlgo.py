import os
import numpy as np
import explainability_techniques.LIME as lime
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

        # create lime explainer
        self.limeExplainer = lime.createLimeExplainer(X)

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

    def optimizeConfidenceStep(self, adaptation, confidence, excludedFeatures):
        # find the most critical requirement
        criticalReq = np.where(confidence == np.min(confidence[np.where(confidence < self.targetConfidence)]))[0][0]

        neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]
        print(neighborIndex)

        maxPoints = np.array([pdp.getMaximalsOfLine(self.pdps[criticalReq][i], neighborIndex)
                              for i in range(len(self.controllableFeatureIndices))])

        usefulMaxPointsIndices = []
        for i, p in enumerate(maxPoints):
            featureIndex = self.controllableFeatureIndices[i]
            if featureIndex not in excludedFeatures and p != adaptation[featureIndex]:
                usefulMaxPointsIndices.append(i)

        if len(usefulMaxPointsIndices) == 0:
            return None, confidence

        controllableIndex = usefulMaxPointsIndices[np.argmax(maxPoints[usefulMaxPointsIndices])]
        maxPoint = maxPoints[controllableIndex]

        featureIndex = self.controllableFeatureIndices[controllableIndex]
        newAdaptation = np.copy(adaptation)
        newAdaptation[featureIndex] = maxPoint
        #excludedFeatures.append(featureIndex)

        newConfidence = vecPredictProba(self.reqClassifiers, [newAdaptation])[0]

        # return if no feature can be modified
        if newAdaptation is None:
            return None, confidence

        print("Confidence gain: " + str(newConfidence - confidence))
        return newAdaptation, newConfidence

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

            excludedFeatures = []
            while len(excludedFeatures) < n_controllableFeatures:
                # recalculate neighbor after the first step
                if len(excludedFeatures) > 0:
                    neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]

                maxYVals = [pdp.getMaxOfLine(self.summaryPdps[j], neighborIndex) for j in range(n_controllableFeatures)]

                maxY = 0
                index = 0
                for j, y in enumerate(maxYVals):
                    if y > maxY:
                        maxY = y
                        index = j
                excludedFeatures.append(index)

                maximals = pdp.getMaximalsOfLine(self.summaryPdps[index], neighborIndex)
                if self.optimizationDirections[index] == -1:
                    # leftmost maximal
                    x = maximals[0]
                else:
                    # rightmost maximal
                    x = maximals[len(maximals) - 1]

                adaptation[self.controllableFeatureIndices[index]] = x

            adaptations[i] = adaptation

        # remove duplicate solutions
        adaptations = np.unique(adaptations, axis=0)

        for adaptation in adaptations:
            neighborIndex = np.ravel(self.knn.kneighbors([adaptation], 1, False))[0]

            maximals = [pdp.getMaximalsOfLine(self.summaryPdps[i], neighborIndex) for i in
                        range(n_controllableFeatures)]

            """
            possibilities = np.array([[maximals[0][i], maximals[1][j], maximals[2][k], maximals[3][n]]
                                      for i in range(len(maximals[0]))
                                      for j in range(len(maximals[1]))
                                      for k in range(len(maximals[2]))
                                      for n in range(len(maximals[3]))])
            print(len(possibilities))
            p = vecPredictProba(self.reqClassifiers,
                                np.append(possibilities,
                                          np.repeat([row[n_controllableFeatures:]], possibilities.shape[0], axis=0),
                                          axis=1))
            print(np.max(p))
            """

            # this should do the magic in most cases,
            # but there could be better solutions not taken into account (see above)
            # we can consider to evaluate more possible solutions
            firstMaximals = []
            for i in range(n_controllableFeatures):
                firstMaximals.append((i, maximals[i][0]))

            additionalAdaptations = []

            def genAdditionalAdaptations(maximals):
                if len(maximals) == 0:
                    return
                for i, m in enumerate(maximals):
                    additionalAdaptation = np.copy(adaptation)
                    additionalAdaptation[self.controllableFeatureIndices[int(m[0])]] = m[1]
                    additionalAdaptations.append(additionalAdaptation)
                    newMaximals = maximals.copy()
                    newMaximals.remove(m)
                    genAdditionalAdaptations(newMaximals)

            genAdditionalAdaptations(firstMaximals)

            """
            print("\nAdditional adaptations:")
            print(np.array(additionalAdaptations)[:, :n_controllableFeatures])
            print("\nAdditional adaptations confidence:")
            print(vecPredictProba(self.reqClassifiers, additionalAdaptations))
            """

            adaptations = np.append(adaptations, additionalAdaptations, axis=0)

        # remove duplicate solutions again
        adaptations = np.unique(adaptations, axis=0)

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
            bestAdaptationIndex = np.argmax(validAdaptationsScores)

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
            bestAdaptationIndex = np.argmax(adaptationsScores)

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
            adaptation = None
            """
            while adaptation is not None and (confidence < self.targetConfidence).any():
                adaptation, confidence = self.optimizeConfidenceStep(adaptation, confidence, excludedFeatures)
                if adaptation is not None: print(adaptation[:n_controllableFeatures])
                print(confidence)
                print()
                confidenceSteps += 1
            """

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
