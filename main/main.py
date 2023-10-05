import random
import sys
import os
import time
import warnings
from colorama import Fore, Style

import pandas as pd
import numpy as np
import explainability_techniques.LIME as lime
import explainability_techniques.PDP as pdp
from sklearn.model_selection import train_test_split
from model.ModelConstructor import constructModel
from genetic_algorithm.NSGA3 import nsga3
from sklearn.neighbors import KNeighborsClassifier


class Req:
    def __init__(self, name):
        self.name = name


def vecPredictProba(models, X):
    probas = []
    for model in models:
        probas.append(model.predict_proba(X))
    probas = np.ravel(probas)[1::2]
    probas = np.column_stack(np.split(probas, n_reqs))
    return probas


def score(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])


if __name__ == '__main__':
    os.chdir(sys.path[0])
    # suppress all warnings
    warnings.filterwarnings("ignore")

    ds = pd.read_csv('../datasets/dataset5000.csv')
    featureNames = ["cruise speed",
                    "image resolution",
                    "illuminance",
                    "controls responsiveness",
                    "power",
                    "smoke intensity",
                    "obstacle size",
                    "obstacle distance",
                    "firm obstacle"]
    controllableFeaturesNames = featureNames[0:4]
    externalFeaturesNames = featureNames[4:9]
    reqNames = ["req_0", "req_1", "req_2", "req_3"]

    # establishes if the controllable features must be minimized (-1) or maximized (1)
    optimizationDirection = [1, -1, -1, -1]

    reqs = [Req(reqNames[0])]

    n_reqs = len(reqs)
    n_neighbors = 10
    n_controllableFeatures = len(controllableFeaturesNames)

    targetConfidence = np.full((1, n_reqs), 0.8)

    # split the dataset
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqNames]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # train a k nearest neighbors classifier only used to find the neighbors of a sample in the dataset
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, np.zeros((X_train.shape[0], 1)))

    for req in reqs:
        print(Fore.RED + "Requirement: " + req.name + "\n" + Style.RESET_ALL)

        req.model = constructModel(X_train.values,
                                   X_test.values,
                                   np.ravel(y_train.loc[:, req.name]),
                                   np.ravel(y_test.loc[:, req.name]))
        print("=" * 100)

        # make pdps
        req.pdps = {}
        for i, f in enumerate(controllableFeaturesNames):
            path = '../plots/' + req.name + '/individuals'
            if not os.path.exists(path):
                os.makedirs(path)
            req.pdps[i] = pdp.partialDependencePlot(req.model, X_train, [f], "both", path + '/' + f + '.png')

        # create lime explainer
        req.limeExplainer = lime.createLimeExplainer(X_train)

    models = []
    for req in reqs:
        models.append(req.model)

    # make summary pdps
    path = "../plots/summary/individuals"
    if not os.path.exists(path):
        os.makedirs(path)
    pdps = {}
    summaryPdps = []
    for i, f in enumerate(controllableFeaturesNames):
        pdps[i] = []
        for req in reqs:
            pdps[i].append(req.pdps[i])
        summaryPdps.append(pdp.multiplyPdps(pdps[i], path + "/" + f + ".png"))

    # metrics
    meanCustomScore = 0
    meanNSGA3Score = 0
    meanSpeedup = 0             # custom vs nsga3
    meanScoreDiff = 0           # custom vs nsga3
    meanDeeperTimeDiff = 0      # deeper vs custom
    meanDeeperScoreDiff = 0     # deeper vs custom
    failedAdaptations = 0

    # adaptations
    results = []
    deeperSearch = True

    # pdp max points of mean line can be computed a priori
    """
    meanLineMaxPoints = {}
    for req in reqs:
        meanLineMaxPoints[req.name] = []
        for i in range(4):
            meanLineMaxPoints[req.name].append(pdp.getMaxPointOfMeanLine(req.pdps[i]))
    """

    variableMin = 0
    variableMax = 100
    variableDomainSize = variableMax - variableMin

    yDeltaMin = 1/100
    deltaMax = variableDomainSize/20

    path = "../plots/adaptation/"
    if not os.path.exists(path):
        os.makedirs(path)

    testNum = 20
    for k in range(1, testNum + 1):
        random.seed()
        rowIndex = k - 1  # random.randrange(0, X_train.shape[0])
        row = X_train.iloc[rowIndex, :].to_numpy()  # TODO use validation set instead of the training one

        print(Fore.BLUE + "Test " + str(k) + ":" + Style.RESET_ALL)
        print("Row " + str(rowIndex) + ":\n" + str(row))
        print("-" * 100)

        for req in reqs:
            lime.saveExplanation(lime.explain(req.limeExplainer, req.model, row), path + req.name + "_starting")

        startTime = time.time()

        print(neigh.kneighbors([row], n_neighbors))
        neighbors = np.ravel(neigh.kneighbors([row], n_neighbors, False))

        # starting solutions
        adaptations = np.empty((n_neighbors, len(row)))
        for i in range(n_neighbors):
            neighborIndex = neighbors[i]
            adaptation = np.copy(row)

            # TODO make this reliable: the adaptation changes randomly based on the order in which features are modified
            # this can be done multiple times but it doesn't converge (I tried)
            for n in range(10):
                controllableFeatures = list(range(n_controllableFeatures))
                while len(controllableFeatures) > 0:
                    featureIndex = random.choice(controllableFeatures)
                    adaptation[featureIndex] = pdp.getMaxPointOfLine(summaryPdps[featureIndex], neighborIndex)
                    neighborIndex = np.ravel(neigh.kneighbors([adaptation], 1, False))[0]
                    controllableFeatures.remove(featureIndex)
            adaptations[i] = adaptation

        adaptationsConfidence = vecPredictProba(models, adaptations)

        print(adaptations[:, :n_controllableFeatures])
        print(adaptationsConfidence)

        validAdaptations = []
        for i, confidence in enumerate(adaptationsConfidence):
            if (confidence >= targetConfidence).all():
                validAdaptations.append(i)

        adaptations = adaptations[validAdaptations]
        adaptationsConfidence = adaptationsConfidence[validAdaptations]

        # enhanced solutions
        adaptationsSteps = []
        for n in range(len(adaptations)):
            step = 0
            excludedFeatures = []
            adaptation = adaptations[n]
            confidence = adaptationsConfidence[n]
            lastAdaptation = np.copy(adaptation)
            lastConfidence = np.copy(confidence)
            while (lastConfidence >= targetConfidence).all() or len(excludedFeatures) < n_controllableFeatures:
                # select the next feature to modify
                featureIndex = None
                bestIncrement = None
                bestSlope = None
                for i in range(n_controllableFeatures):
                    if i not in excludedFeatures:
                        slope = pdp.getSlopeOfClosestLine(summaryPdps[i], adaptation[i], np.prod(lastConfidence))
                        increment = slope * optimizationDirection[i]
                        if bestIncrement is None or increment > bestIncrement:
                            featureIndex = i
                            bestSlope = slope
                            bestIncrement = increment

                # stop if no feature can be improved
                if featureIndex is None:
                    break
                # print(featureIndex)

                # modify the selected feature
                # print("slope: " + str(bestSlope))
                # yDelta = max((lastProbas - targetProbas) * 1.1 / 4, yDeltaMin)
                # print("yDelta :" + str(yDelta))
                delta = 1  # min(yDelta / slope, deltaMax)
                # print("delta: " + str(delta))
                # print("before modification: " + str(lastAdaptation[featureIndex]))
                lastAdaptation[featureIndex] += optimizationDirection[featureIndex] * delta

                if lastAdaptation[featureIndex] < variableMin:
                    lastAdaptation[featureIndex] = variableMin
                    excludedFeatures.append(featureIndex)
                elif lastAdaptation[featureIndex] > variableMax:
                    lastAdaptation[featureIndex] = variableMax
                    excludedFeatures.append(featureIndex)

                # print("after modification: " + str(lastAdaptation[featureIndex]))
                # print(lastAdaptation[0:len(controllableFeaturesNames)])
                # print(excludedFeatures)

                lastConfidence = vecPredictProba(models, [lastAdaptation])
                # print("proba: " + str(lastProbas) + "\n")

                if (lastConfidence < targetConfidence).any():
                    lastAdaptation = np.copy(adaptation)
                    lastConfidence = np.copy(confidence)
                    excludedFeatures.append(featureIndex)
                    # print("discarded\n")
                else:
                    adaptation = np.copy(lastAdaptation)
                    confidence = np.copy(lastConfidence)
                    # print("accepted\n")

                step += 1

            adaptations[n] = adaptation
            adaptationsConfidence[n] = confidence
            adaptationsSteps.append(step)

        endTime = time.time()
        customTime = endTime - startTime

        if len(adaptations) > 0:
            scores = [score(a) for a in adaptations]
            customScore = np.max(scores)
            customAdaptationIndex = np.where(scores == customScore)[0][0]
            customAdaptation = adaptations[customAdaptationIndex]
            customConfidence = adaptationsConfidence[customAdaptationIndex]

            for req in reqs:
                lime.saveExplanation(lime.explain(req.limeExplainer, req.model, row), path + req.name + "_final")

            print("Best adaptation: " + str(customAdaptation[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence))
            print("Adaptation score:                " + str(customScore) + " / 400")
            print("Total steps:                     " + str(adaptationsSteps[customAdaptationIndex]))
        else:
            print("No adaptation found")
            customAdaptation = None
            customConfidence = None
            customScore = None

        print("Custom algorithm execution time: " + str(customTime) + " s")
        print("-" * 100)

        """
        # deeper optimization algorithm
        startTime = time.time()

        step = 0
        deltaScore = 100
        increment = 1
        increaseDecrement = increment/50
        treshold = increment/100

        lastProba = model.predict_proba([adaptation])[0, 1]
        if deeperSearch and custom_proba > targetProba:
            while deltaScore > treshold and lastProba > targetProba:
                slopes = [pdp.getSlopeOfClosestLine(pdps[i], adaptation[i], lastProba) for i in range(len(controllableFeaturesNames))]
                # print("slopes: " + str(slopes))
                increments = [-s * d for s,d in zip(slopes,optimizationDirection)]
                # print("increments: " + str(increments))
                featureToOptimize = increments.index(min(increments))
                # print("featureToOptimize: " + str(featureToOptimize))
                featureToCompromise = increments.index(max(increments))
                # print("featureToCompromise: " + str(featureToCompromise))

                scoreBefore = 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])

                lastAdaptation[featureToOptimize] += increment * optimizationDirection[featureToOptimize]

                if lastAdaptation[featureToOptimize] < variableMin:
                    lastAdaptation[featureToOptimize] = variableMin
                elif lastAdaptation[featureToOptimize] > variableMax:
                    lastAdaptation[featureToOptimize] = variableMax

                decrement = abs(increments[featureToOptimize]/increments[featureToCompromise])
                lastProba = model.predict_proba([lastAdaptation])[0, 1]
                while lastProba < targetProba and abs(decrement) < increment:
                    lastAdaptation[featureToCompromise] = adaptation[featureToCompromise]
                    lastAdaptation[featureToCompromise] -= decrement * optimizationDirection[featureToCompromise]

                    if lastAdaptation[featureToCompromise] < variableMin:
                        lastAdaptation[featureToCompromise] = variableMin
                    elif lastAdaptation[featureToCompromise] > variableMax:
                        lastAdaptation[featureToCompromise] = variableMax

                    lastProba = model.predict_proba([lastAdaptation])[0, 1]
                    # print("proba: " + str(lastProba))

                    # print("decrement: " + str(decrement))
                    decrement += increaseDecrement

                # print(adaptation[0:4])
                # print(lastAdaptation[0:4])
                # print([l - a for a, l in zip(adaptation, lastAdaptation)][0:4])

                scoreAfter = 400 - (100 - lastAdaptation[0] + lastAdaptation[1] + lastAdaptation[2] + lastAdaptation[3])
                # print("Score: " + str(scoreAfter))
                deltaScore = scoreAfter - scoreBefore
                # print("deltaScore: " + str(deltaScore))
                # print("proba: " + str(lastProba) + "\n")

                if lastProba < targetProba or deltaScore < 0:
                    lastAdaptation = np.copy(adaptation)
                    # print("discarded\n")
                else:
                    adaptation = np.copy(lastAdaptation)
                    # print("accepted\n")
                step += 1

            endTime = time.time()
            deeperAlgoTime = endTime - startTime

        # if lastProba > targetProba:
            deeperAdaptation = np.copy(adaptation)
            print("Deeper Adaptation:")
            print(deeperAdaptation[0:4])
            deeper_proba = model.predict_proba([deeperAdaptation])[0, 1]
            print("Model confidence:\t" + str(deeper_proba))
            deeperAlgoScore = 400 - (100 - deeperAdaptation[0] + deeperAdaptation[1] + deeperAdaptation[2] + deeperAdaptation[3])
            deeperScoreImprovement = deeperAlgoScore - customAlgoScore
            deeperScoreImprovementPerc = "{:.2%}".format(deeperScoreImprovement / customAlgoScore)
            meanScoreImprovement = (meanScoreImprovement * (k - 1) + deeperScoreImprovement / customAlgoScore) / k
            meanScoreImprovementPerc = "{:.2%}".format(meanScoreImprovement)
            print("Score improvement:\t" + str(deeperScoreImprovement))
            print("Score improvement[%]:\t" + deeperScoreImprovementPerc)
            print("Total steps:\t" + str(step))
            lime.saveExplanation(lime.explain(explainer, model, deeperAdaptation), path + "final")

            print("\nDeeper algorithm execution time: " + str(deeperAlgoTime) + " s")
            print("\nDeeper algorithm time addition[%]: +" + "{:.2%}".format(deeperAlgoTime / customTime))
            print("-------------------------------------------------------------------------------------------------------")
        else:
            deeperAlgoTime = 0
            deeperAdaptation = None
            deeper_proba = None
            deeperAlgoScore = None
            deeperScoreImprovement = 0
            deeperScoreImprovementPerc = None
        """
        deeperTime = 0
        deeperAdaptation = None
        deeperConfidence = None
        deeperScore = None
        deeperScoreDiff = 0
        deeperScoreImprovement = 0

        # genetic algorithm
        externalFeatures = row[n_controllableFeatures:]

        startTime = time.time()
        res = nsga3(models, targetConfidence, externalFeatures)
        endTime = time.time()
        nsga3Time = endTime - startTime

        """
        print("\nPossible adaptations:")
        print(res.X)

        if res.X is not None:
            print("\nModel confidence:")
            xFull = np.c_[res.X, np.tile(constantFeatures, (res.X.shape[0], 1))]
            print(model.predict_proba(xFull)[:, 1])

        if res.X is not None:
            print("\nScores:")
            print(scores)
            print("Best score: " + str(np.max(scores)))

        print("\nNSGA3 execution time: " + str(nsga3Time) + " s")

        if customTime != 0:
            print("\nSpeed-up: " + str(nsga3Time / customTime) + "x")
        """

        if res.X is not None:
            scores = [score(a) for a in res.X]
            nsga3Score = np.max(scores)
            nsga3AdaptationIndex = np.where(scores == nsga3Score)
            nsga3Adaptation = res.X[nsga3AdaptationIndex][0]
            nsga3Confidence = vecPredictProba(models, [np.append(nsga3Adaptation, externalFeatures)])

            print("Best NSGA3 adaptation:           " + str(nsga3Adaptation))
            print("Model confidence:                " + str(nsga3Confidence))
            print("Adaptation score:                " + str(nsga3Score) + " / 400")
        else:
            print("No adaptation found")
            nsga3Adaptation = None
            nsga3Confidence = None
            nsga3Score = None

        print("NSGA3 execution time:                " + str(nsga3Time) + " s")

        print("-" * 100)

        scoreDiff = None
        scoreImprovement = None

        speedup = nsga3Time / (customTime + deeperTime)
        meanSpeedup = (meanSpeedup * (k - 1) + speedup) / k
        print(Fore.GREEN + "(custom vs nsga3)  Speed-up: " + " " * 14 + str(speedup) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            scoreDiff = customScore + deeperScoreImprovement - nsga3Score
            scoreImprovement = scoreDiff / nsga3Score
            print("(custom vs nsga3)  Score diff:        " + " " * 5 + str(scoreDiff))
            print("(custom vs nsga3)  Score improvement: " + " " * 5 + "{:.2%}".format(scoreImprovement))
        else:
            failedAdaptations += 1

        print(Style.RESET_ALL + Fore.YELLOW + "(custom vs nsga3)  Mean speed-up: " + " " * 9 + str(meanSpeedup) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptations) + customScore) / (k - failedAdaptations)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptations) + nsga3Score) / (k - failedAdaptations)
            meanScoreDiff = (meanScoreDiff * (k - 1 - failedAdaptations) + scoreDiff) / (k - failedAdaptations)
            meanDeeperScoreDiff = (meanDeeperScoreDiff * (k - 1 - failedAdaptations) + deeperScoreDiff) / (k - failedAdaptations)
            meanScoreImprovement = meanScoreDiff / meanNSGA3Score
            meanDeeperScoreImprovement = meanDeeperScoreDiff / meanCustomScore
            print("(custom vs nsga3)  Mean score diff:        " + str(meanScoreDiff))
            print("(custom vs nsga3)  Mean score improvement: " + "{:.2%}".format(meanScoreImprovement))
            print("(deeper vs custom) Mean score diff:        " + str(meanDeeperScoreDiff))
            print("(deeper vs custom) Mean score improvement: " + "{:.2%}".format(meanDeeperScoreImprovement))

        print(Style.RESET_ALL + "=" * 100)

        results.append([nsga3Adaptation, customAdaptation, deeperAdaptation,
                        nsga3Confidence, customConfidence, deeperConfidence,
                        nsga3Score, customScore, deeperScore, deeperScoreImprovement, scoreDiff, scoreImprovement,
                        nsga3Time, customTime, deeperTime, speedup])

    results = pd.DataFrame(results, columns=["nsga3_adaptation", "custom_adaptation", "deeper_adaptation",
                                             "nsga3_confidence", "custom_confidence", "deeper_proba",
                                             "nsga3_score", "custom_score", "deeper_score", "deeper_improvement[%]", "score_diff", "score_improvement[%]",
                                             "nsga3_time", "custom_time", "deeper_time", "speed-up"])
    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_csv(path + "/results.csv")

    """
    mod_dataset = X.to_numpy(copy=True)

    for i in range(mod_dataset.shape[0]):
        if y.iloc[i].bool:
            print("Row: " + str(i))
            constantFeatures = mod_dataset[i, 4:9]

            res = nsga3(bestModel, constantFeatures, featureNames)

            if res.X is not None:
                mod_dataset[i, 0:4] = res.X[0]          #choose an option
    """

    """
    couplesOfFeatures = []
    featureToCycles = features.copy()
    for f1 in features:
        featureToCycles.remove(f1)
        for f2 in featureToCycles:
            couplesOfFeatures.append((f1, f2))   

        path = '../plots/' + bestModel.__class__.__name__
        if not os.path.exists(path): os.makedirs(path)
        
    for f in features:
        path = '../plots/' + bestModel.__class__.__name__ + '/individuals'
        if not os.path.exists(path): os.makedirs(path)
        partial_dependence_plot(bestModel, X_train, [f], "both", path + '/' + f + '.png')
    """
    """
    for c in couplesOfFeatures:
        path = '../plots/' + bestModel.__class__.__name__ + '/couples'
        if not os.path.exists(path): os.makedirs(path)
        partial_dependence_plot(bestModel, X_train, [c], "average", path + '/' + c[0] + ' % ' + c[1] + '.png')
    """

    """
    data_row = X_test.iloc[50]
    local_exp = sort_variables_from_LIME(X_train, bestmodel, data_row, features)
    print(local_exp)
    """

    """
    os.chdir("../MDP_Dataset_Builder")
    mod_dataset = X.to_numpy(copy=True)                         #.loc[0:10, :]. to test only part of the dataset
    np.save("./starting_combinations.npy", mod_dataset)
    os.system("execute.bat ./starting_combinations.npy")
    os.system("merge_csvs.py")
    """
