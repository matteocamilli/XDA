from explainability_techniques.FeatureImportance import permutation_importance_classifier
import time
import numpy as np
from CustomPlanner import CustomPlanner
from sklearn.neighbors import KNeighborsClassifier
import explainability_techniques.PDP as pdp
from util import vecPredictProba, cartesian_product
import faiss


class FICustomPlanner(CustomPlanner):

    def __init__(self, X, Y, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                 controllableFeaturesNames,
                 controllableFeatureIndices, controllableFeatureDomains, optimizationDirections,
                 optimizationScoreFunction, delta=1, plotsPath=None):
        super().__init__(X, n_neighbors, n_startingSolutions, reqClassifiers, targetConfidence,
                         controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                         optimizationDirections, optimizationScoreFunction, delta, plotsPath)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        self.nlist = 10
        self.index = faiss.IndexIVFFlat(self.quantizer, X.shape[1], self.nlist)
        self.index.train(X)
        self.index.add(X)

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
        self.rankingTime = endTime - startTime
        print("FI classifier duration: " + str(endTime - startTime) + " s")
        print("=" * 100)

    def optimizeScoreStep(self, adaptation, confidence, isValidAdaptation, neighborIndex, excludedFeatures,
                          tempExcludedFeatures):

        featureIndices = [i for i in self.controllableFeatureIndices if
                          i not in excludedFeatures and i not in tempExcludedFeatures]

        if not featureIndices:
            return None, None

        newAdaptation = np.copy(adaptation)

        for featureIndex in featureIndices:
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
            tempExcludedFeatures.extend(featureIndices)
        else:
            tempExcludedFeatures.clear()

        return newAdaptation, newConfidence

    def findAdaptation(self, row):
        n_controllableFeatures = len(self.controllableFeatureIndices)

        distances, neighbors = self.index.search(x=np.expand_dims(row, axis=0), k=self.n_neighbors)

        # starting solutions
        adaptations = [row]
        for i in range(self.n_neighbors):
            neighborIndex = neighbors[0][i]
            adaptation = np.copy(row)

            recalculateNeighbor = False
            excludedFeatures = []
            while len(excludedFeatures) < n_controllableFeatures:
                # recalculate neighbor after the first step
                if recalculateNeighbor:
                    distances, neighborIndex = self.index.search(x=np.expand_dims(adaptation, axis=0), k=1)
                    neighborIndex = neighborIndex[0][0]
                recalculateNeighbor = True

                for controllableIndex in self.controllableFeatureIndices:
                    if controllableIndex not in excludedFeatures:
                        maximals = pdp.getMaximalsOfLine(self.summaryPdps[controllableIndex], neighborIndex)
                        if self.optimizationDirections[controllableIndex] == -1:
                            # leftmost maximal
                            x = maximals[0]
                        else:
                            # rightmost maximal
                            x = maximals[len(maximals) - 1]

                    newAdaptation = np.copy(adaptation)
                    newAdaptation[controllableIndex] = x
                    adaptations.append(newAdaptation)
                    excludedFeatures.append(controllableIndex)

        # remove duplicate solutions
        adaptations = np.unique(adaptations, axis=0)

        for adaptation in adaptations:
            distances, neighborIndex = self.index.search(x=np.expand_dims(adaptation, axis=0), k=1)
            neighborIndex = neighborIndex[0][0]

            maximals = [0] * len(self.controllableFeatureIndices)
            for i in self.controllableFeatureIndices:
                maximals[i] = pdp.getMaximalsOfLine(self.summaryPdps[i], neighborIndex)

            maxPossibilities = 1000
            n_possibilities = np.prod([len(m) for m in maximals])

            while n_possibilities > maxPossibilities:
                i = np.argmax([len(m) for m in maximals])
                maximals[i] = maximals[i][1::2]
                n_possibilities = np.prod([len(m) for m in maximals])

            possibilities = cartesian_product(*maximals)
            possibilities = np.append(possibilities,
                                      np.repeat([row[self.externalFeatureIndices]], possibilities.shape[0], axis=0),
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
        print(adaptations[:, self.controllableFeatureIndices])
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

        if validAdaptationFound and len(validAdaptations) > self.n_startingSolutions:
            # pick the best n solutions:
            # rank solutions based on success proba and score
            def solutionRank(a, c):
                return (self.optimizationScoreFunction(a) +
                        np.sum(c - self.targetConfidence) / np.sum(solutionRank.ones - self.targetConfidence) * 100)

            # function constant to avoid useless computation at each call
            solutionRank.ones = np.ravel(np.ones((1, len(self.reqClassifiers))))

            validAdaptationsRanks = [solutionRank(validAdaptations[i], validAdaptationsConfidence[i])
                                     for i in range(len(validAdaptations))]

            bestAdaptationIndices = np.argpartition(validAdaptationsRanks, -self.n_startingSolutions)[
                                    -self.n_startingSolutions:]
            bestAdaptations = validAdaptations[bestAdaptationIndices]
            bestAdaptationsConfidence = validAdaptationsConfidence[bestAdaptationIndices]

            """
            print("\nStarting valid adaptations ranking:")
            print(validAdaptationsRanks)
            """
            # print("Best starting valid adaptations: " + str(bestAdaptationIndices))
        elif validAdaptationFound:
            # just keep the starting adaptations
            bestAdaptations = validAdaptations
            bestAdaptationsConfidence = validAdaptationsConfidence
            # print("Starting valid adaptations:")
        else:
            # no valid adaptation found, so:
            # rank solutions based on success proba only
            def solutionRank(c):
                return np.sum(c - self.targetConfidence)

            adaptationsRanks = [solutionRank(adaptationsConfidence[i]) for i in range(len(adaptations))]

            # pick maximum proba solutions
            bestAdaptationIndices = np.where(adaptationsRanks == np.max(adaptationsRanks))
            bestAdaptations = adaptations[bestAdaptationIndices]
            bestAdaptationsConfidence = adaptationsConfidence[bestAdaptationIndices]

            # select the best n based on the score
            if len(bestAdaptations) > self.n_startingSolutions:
                bestAdaptationsScores = [self.optimizationScoreFunction(a) for a in bestAdaptations]
                bestAdaptationIndices = np.argpartition(bestAdaptationsScores, -self.n_startingSolutions)[
                                        -self.n_startingSolutions:]
                bestAdaptations = bestAdaptations[bestAdaptationIndices]
                bestAdaptationsConfidence = bestAdaptationsConfidence[bestAdaptationIndices]

            """
            print("\nStarting adaptations ranking:")
            print(adaptationsRanks)
            """
            # print("Best starting adaptations: " + str(bestAdaptationIndices))

        """
        print(bestAdaptations[:, :n_controllableFeatures])
        print("\nBest starting adaptations confidence:")
        print(bestAdaptationsConfidence)
        """

        # enhance solutions
        optimizationSteps = [0] * len(bestAdaptations)
        for i in range(len(bestAdaptations)):
            calls = 0
            excludedFeatures = []
            tempExcludedFeatures = []
            adaptation = bestAdaptations[i]
            confidence = bestAdaptationsConfidence[i]

            distances, neighborIndex = self.index.search(x=np.expand_dims(adaptation, axis=0), k=1)
            neighborIndex = neighborIndex[0][0]

            while len(excludedFeatures) + len(tempExcludedFeatures) < n_controllableFeatures:
                # recalculate neighbor only once every n function calls lighten the computation
                if calls >= 10:
                    distances, neighborIndex = self.index.search(x=np.expand_dims(adaptation, axis=0), k=1)
                    neighborIndex = neighborIndex[0][0]
                    # print(neighborIndex)
                    calls = 0
                adaptation, confidence = self.optimizeScoreStep(adaptation, confidence, validAdaptationFound,
                                                                neighborIndex, excludedFeatures,
                                                                tempExcludedFeatures)
                optimizationSteps[i] += 1
                calls += 1

            bestAdaptations[i] = adaptation
            bestAdaptationsConfidence[i] = confidence

        # print("\nScore optimization steps: " + str(optimizationSteps))

        # remove duplicate solutions (there can be new duplicates after the optimization phase)
        bestAdaptations, indices = np.unique(bestAdaptations, axis=0, return_index=True)
        bestAdaptationsConfidence = bestAdaptationsConfidence[indices]

        """
        print("\nFinal adaptations:")
        print(bestAdaptations[:, self.controllableFeatureIndices])

        print("\nFinal adaptations confidence:")
        print(bestAdaptationsConfidence)
        """

        bestAdaptationsScores = [self.optimizationScoreFunction(a) for a in bestAdaptations]
        """
        print("\nFinal adaptations scores:")
        print(bestAdaptationsScores)
        """

        finalAdaptationIndex = np.argmax(bestAdaptationsScores)
        finalAdaptation = bestAdaptations[finalAdaptationIndex]
        finalAdaptationConfidence = bestAdaptationsConfidence[finalAdaptationIndex]
        """
        print("\nBest final adaptation: " + str(finalAdaptationIndex))
        print(finalAdaptation[self.controllableFeatureIndices])
        print("With confidence:")
        print(finalAdaptationConfidence)
        print()
        """

        return finalAdaptation, finalAdaptationConfidence, bestAdaptationsScores[finalAdaptationIndex]
