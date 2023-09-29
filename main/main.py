import random
import sys
import math
import os
import time
import requests
from colorama import Fore, Style

import pandas as pd
import numpy as np
import explainability_techniques.LIME as lime
import explainability_techniques.PDP as pdp
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from model.ModelConstructor import constructModel
from genetic_algorithm.NSGA3 import nsga3


class Req:
    def __init__(self, name):
        self.name = name


if __name__ == '__main__':
    os.chdir(sys.path[0])

    ds = pd.read_csv('../datasets/dataset100.csv')
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

    # establishes if the controllable features must be minimized (-1) or maximized (1)
    featureToMinimize = [-1, 1, 1, 1]

    reqs = [Req("req_0"), Req("req_1"), Req("req_2"), Req("req_3")]
    for req in reqs:
        X = ds.loc[:, featureNames]
        y = ds.loc[:, req.name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        y_test = np.ravel(y_test)
        y_train = np.ravel(y_train)

        req.model = constructModel(X_train.values, X_test.values, y_train, y_test)

        # make pdp graphs
        req.pdps = {}
        for i, f in enumerate(controllableFeaturesNames):
            path = '../plots/' + req.name + '/individuals'
            if not os.path.exists(path):
                os.makedirs(path)
            req.pdps[i] = pdp.partialDependencePlot(req.model, X_train, [f], "both", path + '/' + f + '.png')

        # create lime explainer
        req.limeExplainer = lime.createLimeExplainer(X_train)

    meanSpeedup = 0
    meanScoreDiff = 0

    # adaptations

    # pdp max points of mean line can be computed a priori
    meanLineMaxPoints = {}
    for req in reqs:
        meanLineMaxPoints[req.name] = []
        for i in range(4):
            meanLineMaxPoints[req.name].append(pdp.getMaxPointOfMeanLine(req.pdps[i]))

    targetProba = 0.8

    variableMin = 0
    variableMax = 100
    variableDomainSize = variableMax - variableMin

    delta = 1

    for k in range(1, 21):
        print("Test " + str(k) + ":")

        random.seed()
        req = reqs[random.randrange(0, len(reqs))]
        print("Req: " + str(req.name))

        rowIndex = random.randrange(0, X.shape[0])
        print("Row " + str(rowIndex))
        row = X.iloc[rowIndex, :].to_numpy()

        model = req.model
        explainer = req.limeExplainer
        pdps = req.pdps

        path = "../plots/adaptation/"
        if not os.path.exists(path):
            os.makedirs(path)

        print(str(row) + "\n")
        lime.saveExplanation(lime.explain(explainer, model, row), path + "starting")

        """
        # pdp max points of closest line must be computed at each adaptation if needed
        closestLineMaxPoints = []
        for i in range(4):
            closestLineMaxPoints.append(pdp.getMaxPointOfClosestLine(req.pdps[i], row[i], model.predict_proba([row])[0, 1]))
        """

        # max probability solution
        adaptation = np.copy(row)
        for i, best in enumerate(meanLineMaxPoints[req.name]):
            adaptation[i] = best

        lastAdaptation = np.copy(adaptation)

        startTime = time.time()
        step = 0
        excludedFeatures = []
        lastProba = model.predict_proba([lastAdaptation])[0, 1]
        # check if the best possible solution doesn't solve the problem
        if lastProba >= targetProba:
            while lastProba >= targetProba or len(excludedFeatures) < len(controllableFeaturesNames):
                # select the next feature to modify
                explanation = lime.explain(explainer, model, adaptation)
                sortedFeatures = lime.sortExplanation(explanation)

                """
                print("sorted features: " + str(sortedFeatures))
                temp = explanation.local_exp[1].copy()
                temp.sort(key=lambda k: k[0])
                print("explanation: " + str(temp[0:4]))
                """

                featureIndex = -1
                for f in sortedFeatures:
                    # select this feature if it can be improved, otherwise go next
                    index = f[0]
                    if (index < len(controllableFeaturesNames) and index not in excludedFeatures and
                        ((featureToMinimize[index] == -1 and adaptation[index] > variableMin) or
                         (featureToMinimize[index] == 1 and adaptation[index] < variableMax))):
                        featureIndex = index
                        break
                # stop if no feature can be improved
                if featureIndex == -1:
                    break
                # print(featureIndex)
                # modify the selected feature
                slope = abs(pdp.getSlope(pdps[featureIndex], lastAdaptation[featureIndex]))
                # print("slope: " + str(slope))
                lastAdaptation[featureIndex] -= featureToMinimize[featureIndex] * delta / (slope ** (1/5))

                if lastAdaptation[featureIndex] < variableMin:
                    lastAdaptation[featureIndex] = variableMin
                elif lastAdaptation[featureIndex] > variableMax:
                    lastAdaptation[featureIndex] = variableMax

                # print(lastAdaptation[0:len(controllableFeaturesNames)])

                lastProba = model.predict_proba([lastAdaptation])[0, 1]
                # print("proba: " + str(lastProba) + "\n")

                if lastProba < targetProba:
                    excludedFeatures.append(featureIndex)
                else:
                    adaptation = np.copy(lastAdaptation)

                step += 1
        endTime = time.time()
        customTime = endTime - startTime

        print("Adaptation:\t" + str(adaptation))
        print("Model confidence:\t" + str(model.predict_proba([adaptation])[0, 1]))
        customAlgoScore = 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])
        print("Score of the adaptation:\t" + str(customAlgoScore) + "/400")
        print("Total steps:\t" + str(step))
        lime.saveExplanation(lime.explain(explainer, model, adaptation), path + "final")

        print("\nCustom algorithm execution time: " + str(customTime) + " s")

        constantFeatures = X.iloc[rowIndex, 4:9]
        startTime = time.time()
        res = nsga3(model, constantFeatures)
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
            scores = np.array([400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3]) for adaptation in res.X])
            nsga3Score = np.max(scores)
            bestAdaptationIndex = np.where(scores == nsga3Score)

            print("\nBest NSGA3 adaptation:")
            print(res.X[bestAdaptationIndex])

            print("\nModel confidence: " + str(model.predict_proba([np.append(res.X[bestAdaptationIndex], constantFeatures)])[:, 1]))

            print("Score: " + str(nsga3Score))

        print("\nNSGA3 execution time: " + str(nsga3Time) + " s\n")

        if step != 0:
            speedup = nsga3Time / customTime
            scoreDiff = customAlgoScore - nsga3Score
            print(Fore.GREEN + "Speed-up: " + str(speedup) + "x")
            print("Score diff: " + str(scoreDiff) + Style.RESET_ALL)

            meanSpeedup = (meanSpeedup * (k - 1) + speedup) / k
            meanScoreDiff = (meanScoreDiff * (k - 1) + scoreDiff) / k
            print(Fore.YELLOW + "Mean speed-up: " + str(meanSpeedup) + "x")
            print("Mean score diff: " + str(meanScoreDiff) + "\n" + Style.RESET_ALL)
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
