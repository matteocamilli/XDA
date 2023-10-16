import sys
import os
import glob
import time
import warnings
import random
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from model.ModelConstructor import constructModel
import explainability_techniques.LIME as lime
from CustomPlanner import CustomPlanner
from NSGA3Planner import NSGA3Planner
from util import *


# success score function (based on the signed distance with respect to the target success probabilities)
def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)


# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])


if __name__ == '__main__':
    programStartTime = time.time()

    os.chdir(sys.path[0])

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # evaluate adaptations
    evaluate = False

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

    # establishes if the controllable features must be minimized (-1) or maximized (1)
    optimizationDirections = [1, -1, -1, -1]

    reqs = ["req_0", "req_1", "req_2", "req_3"]

    n_reqs = len(reqs)
    n_neighbors = 10
    n_startingSolutions = 10
    n_controllableFeatures = len(controllableFeaturesNames)

    targetConfidence = np.full((1, n_reqs), 0.8)[0]

    # split the dataset
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqs]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    models = []
    for req in reqs:
        print(Fore.RED + "Requirement: " + req + "\n" + Style.RESET_ALL)

        models.append(constructModel(X_train.values,
                                     X_test.values,
                                     np.ravel(y_train.loc[:, req]),
                                     np.ravel(y_test.loc[:, req])))
        print("=" * 100)

    controllableFeatureDomains = np.repeat([[0, 100]], n_controllableFeatures, 0)

    # initialize planners
    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, [0, 1, 2, 3], controllableFeatureDomains,
                                  optimizationDirections, optimizationScore, 1, "../plots")

    nsga3Planner = NSGA3Planner(models, targetConfidence, successScore, optimizationScore)

    # create lime explainer
    limeExplainer = lime.createLimeExplainer(X_train)

    # metrics
    meanCustomScore = 0
    meanNSGA3Score = 0
    meanSpeedup = 0
    meanScoreDiff = 0
    failedAdaptations = 0

    # adaptations
    results = []
    customDataset = []
    nsga3Dataset = []

    path = "../plots/adaptations"
    if not os.path.exists(path):
        os.makedirs(path)

    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)

    testNum = 200
    for k in range(1, testNum + 1):
        random.seed()
        rowIndex = k - 1  # random.randrange(0, X_test.shape[0])
        row = X_test.iloc[rowIndex, :].to_numpy()

        print(Fore.BLUE + "Test " + str(k) + ":" + Style.RESET_ALL)
        print("Row " + str(rowIndex) + ":\n" + str(row))
        print("-" * 100)

        for i, req in enumerate(reqs):
            lime.saveExplanation(lime.explain(limeExplainer, models[i], row),
                                 path + "/" + str(k) + "_" + req + "_starting")

        startTime = time.time()
        customAdaptation, customConfidence, customScore = customPlanner.findAdaptation(row)
        endTime = time.time()
        customTime = endTime - startTime

        if customAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], customAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation:                 " + str(customAdaptation[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence))
            print("Adaptation score:                " + str(customScore) + " / 400")
        else:
            print("No adaptation found")
            customScore = None

        print("Custom algorithm execution time: " + str(customTime) + " s")
        print("-" * 100)

        # genetic algorithm
        externalFeatures = row[n_controllableFeatures:]

        startTime = time.time()
        nsga3Adaptation, nsga3Confidence, nsga3Score = nsga3Planner.findAdaptation(externalFeatures)
        endTime = time.time()
        nsga3Time = endTime - startTime

        print("Best NSGA3 adaptation:           " + str(nsga3Adaptation[:n_controllableFeatures]))
        print("Model confidence:                " + str(nsga3Confidence))
        print("Adaptation score:                " + str(nsga3Score) + " / 400")
        print("NSGA3 execution time:            " + str(nsga3Time) + " s")

        print("-" * 100)

        scoreDiff = None
        scoreImprovement = None

        speedup = nsga3Time / customTime
        meanSpeedup = (meanSpeedup * (k - 1) + speedup) / k
        print(Fore.GREEN + "Speed-up: " + " " * 14 + str(speedup) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            scoreDiff = customScore - nsga3Score
            scoreImprovement = scoreDiff / nsga3Score
            print("Score diff:        " + " " * 5 + str(scoreDiff))
            print("Score improvement: " + " " * 5 + "{:.2%}".format(scoreImprovement))
        else:
            failedAdaptations += 1

        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up: " + " " * 9 + str(meanSpeedup) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptations) + customScore) / (k - failedAdaptations)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptations) + nsga3Score) / (k - failedAdaptations)
            meanScoreDiff = (meanScoreDiff * (k - 1 - failedAdaptations) + scoreDiff) / (k - failedAdaptations)
            meanScoreImprovement = meanScoreDiff / meanNSGA3Score
            print("Mean score diff:        " + str(meanScoreDiff))
            print("Mean score improvement: " + "{:.2%}".format(meanScoreImprovement))

        print(Style.RESET_ALL + "=" * 100)

        results.append([nsga3Adaptation, customAdaptation,
                        nsga3Confidence, customConfidence,
                        nsga3Score, customScore, scoreDiff, scoreImprovement,
                        nsga3Time, customTime, speedup])

    results = pd.DataFrame(results, columns=["nsga3_adaptation", "custom_adaptation",
                                             "nsga3_confidence", "custom_confidence",
                                             "nsga3_score", "custom_score", "score_diff", "score_improvement[%]",
                                             "nsga3_time", "custom_time", "speed-up"])
    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_csv(path + "/results.csv")

    if evaluate:
        evaluateAdaptations(results)

    programEndTime = time.time()
    totalExecutionTime = programEndTime - programStartTime
    print("\nProgram execution time: " + str(totalExecutionTime / 60) + " m")
