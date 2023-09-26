import random
import sys
import math
import os
import time
import requests

import pandas as pd
import numpy as np
import explainability_techniques.LIME as lime
import explainability_techniques.PDP as pdp
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from model.ModelConstructor import constructModel
from genetic_algorithm.NSGA3 import nsga3
from custom_algorithm import skyline_finder

if __name__ == '__main__':

    os.chdir(sys.path[0])
    ds = pd.read_csv('../datasets/dataset500.csv')
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
    outcomes = ["req_1"]  # "req_0", "req_1", "req_2", "req_3", "req_4"
    X = ds.loc[:, featureNames]
    y = ds.loc[:, outcomes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)

    bestModel = constructModel(X_train.values, X_test.values, y_train, y_test)

    # make pdp graphs
    pdps = {}
    for i, f in enumerate(controllableFeaturesNames):
        path = '../plots/' + bestModel.__class__.__name__ + '/individuals'
        if not os.path.exists(path): os.makedirs(path)
        pdps[i] = pdp.partialDependencePlot(bestModel, X_train, [f], "both", path + '/' + f + '.png')

    # adaptations

    # pdp max points can be computed a priori, one for each controllable semantic variable
    maxPoints = []
    for i in range(4):
        maxPoints.append(pdp.getMaxPoint(pdps[i]))

    targetProba = 0.8

    variableMin = 0
    variableMax = 100
    variableDomainSize = variableMax - variableMin

    minDelta = variableDomainSize / 100
    maxDelta = variableDomainSize * 1/4
    factor = maxDelta / targetProba # for normalization and scaling
    delta = minDelta
    precision = variableDomainSize / 60 # just because pdps are lines with 60 points

    # create lime explainer
    explainer = lime.createLIMEExplainer(X_train)

    for k in range(1):
        rowIndex = random.randrange(X.shape[0])
        row = X.iloc[rowIndex, :].to_numpy()

        print(str(row) + "\n")
        lime.printLime(lime.explain(explainer, bestModel, row))

        adaptation = np.copy(row)

        bestSolution = np.copy(row)
        for i, best in enumerate(maxPoints):
            bestSolution[i] = best

        startTime = time.time()
        # check if best solution solve the problem
        if bestModel.predict_proba([bestSolution])[0, 1] < targetProba:
            adaptation = bestSolution
        else:
            lastProba = bestModel.predict_proba([adaptation])[0, 1]
            while lastProba < targetProba:
                # select the next feature to modify
                explanation = lime.explain(explainer, bestModel, adaptation)
                feature_ranked = lime.sortExplanation(explanation)
                feature = feature_ranked[0]
                deltaFromMax = 100
                for f in feature_ranked:
                    feature = f
                    if f[0] < 3:
                        deltaFromMax = abs(adaptation[f[0]] - maxPoints[f[0]])
                        if deltaFromMax > precision: break
                featureIndex = feature[0]
                if deltaFromMax < precision: break
                # modify the selected feature
                adaptation[featureIndex] = adaptation[featureIndex] + np.sign(maxPoints[featureIndex] - adaptation[featureIndex]) * delta
                if adaptation[featureIndex] < variableMin: adaptation[featureIndex] = variableMin
                if adaptation[featureIndex] > variableMax: adaptation[featureIndex] = variableMax
                lastProba = bestModel.predict_proba([adaptation])[0, 1]
                # print(lastProba)
                # calculate the next delta
                delta = max(minDelta, (targetProba - lastProba) * factor)  # just heuristic... maybe it can be done better
        endTime = time.time()
        customTime = endTime - startTime

        print("Adaptation:\t" + str(adaptation))
        print("Model confidence:\t" + str(bestModel.predict_proba([adaptation])[0, 1]))
        score = (400 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])
        print("Score of the adaptation:\t" + str(score) + "/400")
        lime.printLime(lime.explain(explainer, bestModel, adaptation))

        print("\nCustom algorithm execution time: " + str(customTime) + " s")

        constantFeatures = X.iloc[rowIndex, 4:9]
        startTime = time.time()
        res = nsga3(bestModel, constantFeatures, featureNames)
        endTime = time.time()
        nsga3Time = endTime - startTime

        print("\nPossible adaptations:")
        print(res.X)

        if res.X is not None:
            print("\nModel confidence:")
            xFull = np.c_[res.X, np.tile(constantFeatures, (res.X.shape[0], 1))]
            print(bestModel.predict_proba(xFull)[:, 1])

        if res.X is not None:
            print("\nScores:")
            scores = np.array([400 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3] for adaptation in res.X])
            print(scores)
            print("Best score: " + str(np.min(scores)))

        print("\nNSGA3 execution time: " + str(nsga3Time) + " s")

        if customTime is not 0:
            print("\nSpeed-up: " + str(nsga3Time / customTime) + "x")

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
