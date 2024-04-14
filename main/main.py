import sys
import os
import glob
import time
import warnings
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from SHAPCustomPlanner import SHAPCustomPlanner
from PCACustomPlanner import PCACustomPlanner
from FICustomPlanner import FICustomPlanner
from model.ModelConstructor import constructModel
import explainability_techniques.LIME as lime
from CustomPlanner import CustomPlanner
from NSGA3Planner import NSGA3Planner
from util import vecPredictProba, evaluateAdaptations


# success score function (based on the signed distance with respect to the target success probabilities)
def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)


# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])


# ====================================================================================================== #
# IMPORTANT: everything named as custom in the code refers to the XDA approach                           #
#            everything named as confidence in the code refers to the predicted probabilities of success #
# ====================================================================================================== #


if __name__ == '__main__':
    programStartTime = time.time()

    os.chdir(sys.path[0])

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # evaluate adaptations
    evaluate = True

    ds = pd.read_csv('../datasets/drivev2.csv')
    featureNames = ["car_speed",
                    "p_x",
                    "p_y",
                    "orientation",
                    "weather",
                    "road_shape"]
    controllableFeaturesNames = featureNames[0:1]
    externalFeaturesNames = featureNames[1:6]

    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize
    optimizationDirections = [1]

    reqs = ["req_0", "req_1", "req_2"]

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

    controllableFeatureDomains = np.array([[5.0, 50.0]])

    # initialize planners
    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, [0], controllableFeatureDomains,
                                  optimizationDirections, optimizationScore, 1, "../explainability_plots")

    SHAPcustomPlanner = SHAPCustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                          controllableFeaturesNames, [0], controllableFeatureDomains,
                                          optimizationDirections, optimizationScore, 1, "../explainability_plots")

    PCAcustomPlanner = PCACustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                        controllableFeaturesNames, [0], controllableFeatureDomains,
                                        optimizationDirections, optimizationScore, 1, "../explainability_plots")

    FICustomPlanner = FICustomPlanner(X_train, y_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                      controllableFeaturesNames, [0], controllableFeatureDomains,
                                      optimizationDirections, optimizationScore, 1, "../explainability_plots")

    nsga3Planner = NSGA3Planner(models, targetConfidence, [0], controllableFeatureDomains,
                                optimizationDirections, successScore, optimizationScore)

    # create lime explainer
    limeExplainer = lime.createLimeExplainer(X_train)

    # metrics
    meanCustomScore = 0
    meanCustomScoreSHAP = 0
    meanCustomScorePCA = 0
    meanCustomScoreFI = 0
    meanNSGA3Score = 0
    meanSpeedup = 0
    meanSpeedupSHAP = 0
    meanSpeedupPCA = 0
    meanSpeedupFI = 0
    meanScoreDiff = 0
    meanScoreDiffSHAP = 0
    meanScoreDiffPCA = 0
    meanScoreDiffFI = 0
    failedAdaptations = 0
    failedAdaptationsSHAP = 0
    failedAdaptationsPCA = 0
    failedAdaptationsFI = 0

    # adaptations
    results = []
    resultsSHAP = []
    resultsPCA = []
    resultsFI = []
    customDataset = []
    nsga3Dataset = []

    path = "../explainability_plots/adaptations"
    if not os.path.exists(path):
        os.makedirs(path)

    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)

    testNum = 200
    for k in range(1, testNum + 1):
        rowIndex = k - 1
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

        startTime = time.time()
        SHAPcustomAdaptation, SHAPcustomConfidence, SHAPcustomScore = SHAPcustomPlanner.findAdaptation(row)
        endTime = time.time()
        SHAPcustomTime = endTime - startTime

        if SHAPcustomAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], SHAPcustomAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation SHAP:                 " + str(SHAPcustomAdaptation[0:n_controllableFeatures]))
            print("Model confidence SHAP:                " + str(SHAPcustomConfidence))
            print("Adaptation score SHAP:                " + str(SHAPcustomScore) + " / 400")
        else:
            print("No adaptation found")
            SHAPcustomScore = None

        print("Custom SHAP algorithm execution time: " + str(SHAPcustomTime) + " s")
        print("-" * 100)

        startTime = time.time()
        PCAcustomAdaptation, PCAcustomConfidence, PCAcustomScore = PCAcustomPlanner.findAdaptation(row)
        endTime = time.time()
        PCAcustomTime = endTime - startTime

        if PCAcustomAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], PCAcustomAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation PCA:                 " + str(PCAcustomAdaptation[0:n_controllableFeatures]))
            print("Model confidence PCA:                " + str(PCAcustomConfidence))
            print("Adaptation score PCA:                " + str(PCAcustomScore) + " / 400")
        else:
            print("No adaptation found")
            PCAcustomScore = None

        print("Custom PCA algorithm execution time: " + str(PCAcustomTime) + " s")
        print("-" * 100)

        startTime = time.time()
        FIcustomAdaptation, FIcustomConfidence, FIcustomScore = FICustomPlanner.findAdaptation(row)
        endTime = time.time()
        FIcustomTime = endTime - startTime

        if FIcustomAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], FIcustomAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation FI:                 " + str(FIcustomAdaptation[0:n_controllableFeatures]))
            print("Model confidence FI:                " + str(FIcustomConfidence))
            print("Adaptation score FI:                " + str(FIcustomScore) + " / 400")
        else:
            print("No adaptation found")
            FIcustomScore = None

        print("Custom FI algorithm execution time: " + str(FIcustomTime) + " s")
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
        SHAPscoreDiff = None
        PCAscoreDiff = None
        FIscoreDiff = None
        scoreImprovement = None
        SHAPscoreImprovement = None
        PCAscoreImprovement = None
        FIscoreImprovement = None

        speedup = nsga3Time / customTime
        meanSpeedup = (meanSpeedup * (k - 1) + speedup) / k
        print(Fore.GREEN + "Speed-up: " + " " * 14 + str(speedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up Custom: " + " " * 9 + str(meanSpeedup) + "x")
        SHAPspeedup = nsga3Time / SHAPcustomTime
        meanSpeedupSHAP = (meanSpeedupSHAP * (k - 1) + SHAPspeedup) / k
        print(Fore.GREEN + "Speed-up SHAP: " + " " * 14 + str(SHAPspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up SHAP: " + " " * 9 + str(meanSpeedupSHAP) + "x")
        PCAspeedup = nsga3Time / PCAcustomTime
        meanSpeedupPCA = (meanSpeedupPCA * (k - 1) + PCAspeedup) / k
        print(Fore.GREEN + "Speed-up PCA: " + " " * 14 + str(PCAspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up PCA: " + " " * 9 + str(meanSpeedupPCA) + "x")
        FIspeedup = nsga3Time / FIcustomTime
        meanSpeedupFI = (meanSpeedupFI * (k - 1) + FIspeedup) / k
        print(Fore.GREEN + "Speed-up FI: " + " " * 14 + str(FIspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up FI: " + " " * 9 + str(meanSpeedupFI) + "x")

        if customAdaptation is not None and nsga3Adaptation is not None:
            scoreDiff = customScore - nsga3Score
            scoreImprovement = scoreDiff / nsga3Score
            print("Score diff:        " + " " * 5 + str(scoreDiff))
            print("Score improvement: " + " " * 5 + "{:.2%}".format(scoreImprovement))
        else:
            failedAdaptations += 1

        if SHAPcustomAdaptation is not None and nsga3Adaptation is not None:
            SHAPscoreDiff = SHAPcustomScore - nsga3Score
            SHAPscoreImprovement = SHAPscoreDiff / nsga3Score
            print("Score diff SHAP:        " + " " * 5 + str(SHAPscoreDiff))
            print("Score improvement SHAP: " + " " * 5 + "{:.2%}".format(SHAPscoreImprovement))
        else:
            failedAdaptationsSHAP += 1

        if PCAcustomAdaptation is not None and nsga3Adaptation is not None:
            PCAscoreDiff = PCAcustomScore - nsga3Score
            PCAscoreImprovement = PCAscoreDiff / nsga3Score
            print("Score diff PCA:        " + " " * 5 + str(PCAscoreDiff))
            print("Score improvement PCA: " + " " * 5 + "{:.2%}".format(PCAscoreImprovement))
        else:
            failedAdaptationsPCA += 1

        if FIcustomAdaptation is not None and nsga3Adaptation is not None:
            FIscoreDiff = FIcustomScore - nsga3Score
            FIscoreImprovement = FIscoreDiff / nsga3Score
            print("Score diff FI:        " + " " * 5 + str(FIscoreDiff))
            print("Score improvement FI: " + " " * 5 + "{:.2%}".format(FIscoreImprovement))
        else:
            failedAdaptationsFI += 1

        if customAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptations) + customScore) / (k - failedAdaptations)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptations) + nsga3Score) / (k - failedAdaptations)
            meanScoreDiff = (meanScoreDiff * (k - 1 - failedAdaptations) + scoreDiff) / (k - failedAdaptations)
            meanScoreImprovement = meanScoreDiff / meanNSGA3Score
            print("Mean score diff:        " + str(meanScoreDiff))
            print("Mean score improvement: " + "{:.2%}".format(meanScoreImprovement))

        if SHAPcustomAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScoreSHAP = (meanCustomScoreSHAP * (k - 1 - failedAdaptationsSHAP) + SHAPcustomScore) / (
                        k - failedAdaptationsSHAP)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptationsSHAP) + nsga3Score) / (
                        k - failedAdaptationsSHAP)
            meanScoreDiffSHAP = (meanScoreDiffSHAP * (k - 1 - failedAdaptationsSHAP) + scoreDiff) / (
                        k - failedAdaptationsSHAP)
            meanScoreImprovementSHAP = meanScoreDiffSHAP / meanNSGA3Score
            print("Mean score diff SHAP:        " + str(meanScoreDiffSHAP))
            print("Mean score improvement SHAP: " + "{:.2%}".format(meanScoreImprovementSHAP))

        if PCAcustomAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScorePCA = (meanCustomScorePCA * (k - 1 - failedAdaptationsPCA) + PCAcustomScore) / (
                        k - failedAdaptationsPCA)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptationsPCA) + nsga3Score) / (k - failedAdaptationsPCA)
            meanScoreDiffPCA = (meanScoreDiffPCA * (k - 1 - failedAdaptationsPCA) + scoreDiff) / (
                        k - failedAdaptationsPCA)
            meanScoreImprovementPCA = meanScoreDiffPCA / meanNSGA3Score
            print("Mean score diff:        " + str(meanScoreDiffPCA))
            print("Mean score improvement: " + "{:.2%}".format(meanScoreImprovementPCA))

        if FIcustomAdaptation is not None and nsga3Adaptation is not None:
            meanCustomScoreFI = (meanCustomScoreFI * (k - 1 - failedAdaptationsFI) + FIcustomScore) / (
                        k - failedAdaptationsFI)
            meanNSGA3Score = (meanNSGA3Score * (k - 1 - failedAdaptationsFI) + nsga3Score) / (k - failedAdaptationsFI)
            meanScoreDiffFI = (meanScoreDiffFI * (k - 1 - failedAdaptationsFI) + scoreDiff) / (k - failedAdaptationsFI)
            meanScoreImprovementFI = meanScoreDiffPCA / meanNSGA3Score
            print("Mean score diff:        " + str(meanScoreDiffFI))
            print("Mean score improvement: " + "{:.2%}".format(meanScoreImprovementFI))

        print(Style.RESET_ALL + "=" * 100)

        results.append([nsga3Adaptation, customAdaptation,
                        nsga3Confidence, customConfidence,
                        nsga3Score, customScore, scoreDiff, scoreImprovement,
                        nsga3Time, customTime, speedup])

        resultsSHAP.append([nsga3Adaptation, SHAPcustomAdaptation,
                            nsga3Confidence, SHAPcustomConfidence,
                            nsga3Score, SHAPcustomScore, SHAPscoreDiff, SHAPscoreImprovement,
                            nsga3Time, SHAPcustomTime, SHAPspeedup])

        resultsPCA.append([nsga3Adaptation, PCAcustomAdaptation,
                           nsga3Confidence, PCAcustomConfidence,
                           nsga3Score, PCAcustomScore, PCAscoreDiff, PCAscoreImprovement,
                           nsga3Time, PCAcustomTime, PCAspeedup])

        resultsFI.append([nsga3Adaptation, FIcustomAdaptation,
                          nsga3Confidence, FIcustomConfidence,
                          nsga3Score, FIcustomScore, FIscoreDiff, FIscoreImprovement,
                          nsga3Time, FIcustomTime, FIspeedup])

    results = pd.DataFrame(results, columns=["nsga3_adaptation", "custom_adaptation",
                                             "nsga3_confidence", "custom_confidence",
                                             "nsga3_score", "custom_score", "score_diff", "score_improvement[%]",
                                             "nsga3_time", "custom_time", "speed-up"])

    resultsSHAP = pd.DataFrame(resultsSHAP, columns=["nsga3_adaptation", "custom_adaptation",
                                                     "nsga3_confidence", "custom_confidence",
                                                     "nsga3_score", "custom_score", "score_diff",
                                                     "score_improvement[%]",
                                                     "nsga3_time", "custom_time", "speed-up"])

    resultsPCA = pd.DataFrame(resultsPCA, columns=["nsga3_adaptation", "custom_adaptation",
                                                   "nsga3_confidence", "custom_confidence",
                                                   "nsga3_score", "custom_score", "score_diff",
                                                   "score_improvement[%]",
                                                   "nsga3_time", "custom_time", "speed-up"])

    resultsFI = pd.DataFrame(resultsFI, columns=["nsga3_adaptation", "custom_adaptation",
                                                  "nsga3_confidence", "custom_confidence",
                                                  "nsga3_score", "custom_score", "score_diff",
                                                  "score_improvement[%]",
                                                  "nsga3_time", "custom_time", "speed-up"])

    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_csv(path + "/results.csv")
    resultsSHAP.to_csv(path + "/resultsSHAP.csv")
    resultsPCA.to_csv(path + "/resultsPCA.csv")
    resultsFI.to_csv(path + "/resultsFI.csv")

    if evaluate:
        evaluateAdaptations(results, resultsSHAP, resultsPCA, resultsFI, featureNames)

    programEndTime = time.time()
    totalExecutionTime = programEndTime - programStartTime
    print("\nProgram execution time: " + str(totalExecutionTime / 60) + " m")
