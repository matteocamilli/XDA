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
from FICustomPlanner import FICustomPlanner
from model.ModelConstructor import constructModel
import explainability_techniques.LIME as lime
from CustomPlanner import CustomPlanner
from NSGA3Planner import NSGA3Planner
from util import vecPredictProba, evaluateAdaptations
import multilabel_oversampling as mo


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

    ds = pd.read_csv('../datasets/drivev3.csv')
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
        """
        ds_req = pd.read_csv(f'bilanciato_{req}.csv')
        X_req = ds_req.loc[:, featureNames]
        y_req = ds_req.loc[:, req]
        X_req_train, X_req_test, y_req_train, y_req_test = train_test_split(
            X_req, y_req, test_size=0.4, random_state=42
        )
        """
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

    FICustomPlanner = FICustomPlanner(X_train, y_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                      controllableFeaturesNames, [0], controllableFeatureDomains,
                                      optimizationDirections, optimizationScore, 1, "../explainability_plots")

    # create lime explainer
    limeExplainer = lime.createLimeExplainer(X_train)

    # metrics
    meanCustomScore = 0
    meanCustomScoreSHAP = 0
    meanCustomScoreFI = 0
    meanSpeedupSHAP = 0
    meanSpeedupFI = 0
    meanScoreDiffSHAP = 0
    meanScoreDiffFI = 0
    failedAdaptations = 0
    failedAdaptationsSHAP = 0
    failedAdaptationsFI = 0

    # adaptations
    results = []
    resultsSHAP = []
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

        SHAPscoreDiff = None
        FIscoreDiff = None
        SHAPscoreImprovement = None
        FIscoreImprovement = None

        if SHAPcustomTime == 0:
            SHAPspeedup = 0
        else:
            SHAPspeedup = customTime / SHAPcustomTime
        meanSpeedupSHAP = (meanSpeedupSHAP * (k - 1) + SHAPspeedup) / k
        print(Fore.GREEN + "Speed-up SHAP: " + " " * 14 + str(SHAPspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up SHAP: " + " " * 9 + str(meanSpeedupSHAP) + "x")
        if FIcustomTime == 0:
            FIspeedup = 0
        else:
            FIspeedup = customTime / FIcustomTime
        meanSpeedupFI = (meanSpeedupFI * (k - 1) + FIspeedup) / k
        print(Fore.GREEN + "Speed-up FI: " + " " * 14 + str(FIspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up FI: " + " " * 9 + str(meanSpeedupFI) + "x")

        if SHAPcustomAdaptation is not None and customAdaptation is not None:
            SHAPscoreDiff = SHAPcustomScore - customScore
            SHAPscoreImprovement = SHAPscoreDiff / customScore
            print("Score diff SHAP:        " + " " * 5 + str(SHAPscoreDiff))
            print("Score improvement SHAP: " + " " * 5 + "{:.2%}".format(SHAPscoreImprovement))
        else:
            failedAdaptationsSHAP += 1

        if FIcustomAdaptation is not None and customAdaptation is not None:
            FIscoreDiff = FIcustomScore - customScore
            FIscoreImprovement = FIscoreDiff / customScore
            print("Score diff FI:        " + " " * 5 + str(FIscoreDiff))
            print("Score improvement FI: " + " " * 5 + "{:.2%}".format(FIscoreImprovement))
        else:
            failedAdaptationsFI += 1

        if SHAPcustomAdaptation is not None and customAdaptation is not None:
            meanCustomScoreSHAP = (meanCustomScoreSHAP * (k - 1 - failedAdaptationsSHAP) + SHAPcustomScore) / (
                    k - failedAdaptationsSHAP)
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsSHAP) + customScore) / (
                    k - failedAdaptationsSHAP)
            meanScoreDiffSHAP = (meanScoreDiffSHAP * (k - 1 - failedAdaptationsSHAP) + SHAPscoreDiff) / (
                    k - failedAdaptationsSHAP)
            meanScoreImprovementSHAP = meanScoreDiffSHAP / meanCustomScore
            print("Mean score diff SHAP:        " + str(meanScoreDiffSHAP))
            print("Mean score improvement SHAP: " + "{:.2%}".format(meanScoreImprovementSHAP))

        if FIcustomAdaptation is not None and customAdaptation is not None:
            meanCustomScoreFI = (meanCustomScoreFI * (k - 1 - failedAdaptationsFI) + FIcustomScore) / (
                    k - failedAdaptationsFI)
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsFI) + customScore) / (k - failedAdaptationsFI)
            meanScoreDiffFI = (meanScoreDiffFI * (k - 1 - failedAdaptationsFI) + FIscoreDiff) / (k - failedAdaptationsFI)
            meanScoreImprovementFI = meanScoreDiffFI / meanCustomScore
            print("Mean score diff:        " + str(meanScoreDiffFI))
            print("Mean score improvement: " + "{:.2%}".format(meanScoreImprovementFI))

        print(Style.RESET_ALL + "=" * 100)

        results.append([customAdaptation,
                        customConfidence,
                        customScore,
                        customTime])

        resultsSHAP.append([SHAPcustomAdaptation,
                            SHAPcustomConfidence,
                            SHAPcustomScore, SHAPscoreDiff, SHAPscoreImprovement,
                            SHAPcustomTime, SHAPspeedup])

        resultsFI.append([FIcustomAdaptation,
                          FIcustomConfidence,
                          FIcustomScore, FIscoreDiff, FIscoreImprovement,
                          FIcustomTime, FIspeedup])

    results = pd.DataFrame(results, columns=["custom_adaptation",
                                             "custom_confidence",
                                             "custom_score",
                                             "custom_time"])

    resultsSHAP = pd.DataFrame(resultsSHAP, columns=["custom_adaptation",
                                                     "custom_confidence",
                                                     "custom_score", "score_diff",
                                                     "score_improvement[%]",
                                                     "custom_time", "speed-up"])

    resultsFI = pd.DataFrame(resultsFI, columns=["custom_adaptation",
                                                 "custom_confidence",
                                                 "custom_score", "score_diff",
                                                 "score_improvement[%]",
                                                 "custom_time", "speed-up"])

    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)
    results.to_csv(path + "/results.csv")
    resultsSHAP.to_csv(path + "/resultsSHAP.csv")
    resultsFI.to_csv(path + "/resultsFI.csv")

    if evaluate:
        evaluateAdaptations(results, resultsSHAP, resultsFI, featureNames)

    programEndTime = time.time()
    totalExecutionTime = programEndTime - programStartTime
    print("\nProgram execution time: " + str(totalExecutionTime / 60) + " m")
