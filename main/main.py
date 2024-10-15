import sys
import os
import glob
import time
import warnings
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from CustomPlanner import CustomPlanner
from model.ModelConstructor import constructModel
import explainability_techniques.LIME as lime
from NSGA3Planner import NSGA3Planner
from SHAPCustomPlanner import SHAPCustomPlanner
from FICustomPlanner import FICustomPlanner
from util import vecPredictProba, evaluateAdaptations
from FITEST import FitestPlanner
from RandomCustomPlanner import RandomPlanner


# import multilabel_oversampling as mo


# success score function (based on the signed distance with respect to the target success probabilities)
def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)


def normalizeAdaptation(adaptation):
    new_adaptation = []
    for index in range(n_controllableFeatures):
        new_adaptation.append(((adaptation[index] - controllableFeatureDomains[index][0]) / (
                    controllableFeatureDomains[index][1] - controllableFeatureDomains[index][0])) * 100)

    return new_adaptation


# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    adaptation = normalizeAdaptation(adaptation)
    tot = 100 * n_controllableFeatures
    score = 0
    for i in range(n_controllableFeatures):
        if optimizationDirections[i] == 1:
            score += 100 - adaptation[i]
        else:
            score += adaptation[i]
    return tot - score

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

    ds = pd.read_csv('../datasets/uavv2.csv')
    featureNames = ['formation','flying_speed','countermeasure','weather','day_time','threat_range','#threats']
    controllableFeaturesNames = featureNames[0:3]
    externalFeaturesNames = featureNames[3:7]
    controllableFeatureIndices = [0, 1, 2]

    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize
    optimizationDirections = [1, 1, -1]

    reqs = ["req_0", "req_1", "req_2", "req_3", "req_4", "req_5", "req_6", "req_7", "req_8", "req_9", "req_10", "req_11"]

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

    controllableFeatureDomains = np.array([[0, 1], [5.0, 50.0], [0, 1]])
    discreteIndices = []
    # initialize planners

    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                                  optimizationDirections, optimizationScore, 1, "../explainability_plots")

    nsga3Planner = NSGA3Planner(models, targetConfidence, controllableFeatureIndices, controllableFeatureDomains,
                                optimizationDirections, successScore, optimizationScore)

    SHAPcustomPlanner = SHAPCustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                          controllableFeaturesNames, controllableFeatureIndices,
                                          controllableFeatureDomains,
                                          optimizationDirections, optimizationScore, 1, "../explainability_plots")

    FICustomPlanner = FICustomPlanner(X_train, y_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                      controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                                      optimizationDirections, optimizationScore, 1, "../explainability_plots")

    pop_size = nsga3Planner.algorithm.pop_size

    FitestPlanner = FitestPlanner(models, targetConfidence,
                                  controllableFeatureIndices, controllableFeatureDomains, optimizationScore,
                                  successScore,
                                  pop_size,
                                  discreteIndices, 12, [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    RandomPlanner = RandomPlanner(controllableFeatureIndices, controllableFeatureDomains, discreteIndices, models,
                                  optimizationScore)

    # create lime explainer
    limeExplainer = lime.createLimeExplainer(X_train)

    # metrics
    meanCustomScore = 0
    meanCustomScoreSHAP = 0
    meanCustomScoreFI = 0
    meanCustomScoreFitest = 0
    meanCustomScoreRandom = 0
    meanSpeedupSHAP = 0
    meanSpeedupFI = 0
    meanSpeedupFitest = 0
    meanSpeedupRandom = 0
    meanScoreDiffSHAP = 0
    meanScoreDiffFI = 0
    meanScoreDiffFitest = 0
    meanScoreDiffRandom = 0
    failedAdaptations = 0
    failedAdaptationsSHAP = 0
    failedAdaptationsFI = 0
    failedAdaptationsFitest = 0
    failedAdaptationsRandom = 0

    # adaptations
    results = []
    resultsSHAP = []
    resultsFI = []
    resultsFitest = []
    resultsRandom = []
    customDataset = []
    nsga3Dataset = []
    resultsNSGA = []

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
            print("Adaptation score:                " + str(customScore) + " /" + str(100 * n_controllableFeatures))
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
            print("Adaptation score SHAP:                " + str(SHAPcustomScore) + " /" + str(100 * n_controllableFeatures))
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
            print("Adaptation score FI:                " + str(FIcustomScore) + " /" + str(100 * n_controllableFeatures))
        else:
            print("No adaptation found")
            FIcustomScore = None

        print("Custom FI algorithm execution time: " + str(FIcustomTime) + " s")
        print("-" * 100)

        startTime = time.time()
        FitestcustomAdaptation, FitestcustomConfidence, FitestcustomScore = FitestPlanner.run_search(row)
        endTime = time.time()
        FitestcustomTime = endTime - startTime

        if FitestcustomAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], FitestcustomAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation Fitest:                 " + str(FitestcustomAdaptation[0:n_controllableFeatures]))
            print("Model confidence Fitest:                " + str(FitestcustomConfidence))
            print("Adaptation score Fitest:                " + str(FitestcustomScore) + " /" + str(100 * n_controllableFeatures))
        else:
            print("No adaptation found")
            FitestcustomScore = None

        print("Fitest algorithm execution time: " + str(FitestcustomTime) + " s")
        print("-" * 100)

        startTime = time.time()
        RandomCustomAdaptation, RandomCustomConfidence, RandomCustomScore = RandomPlanner.findAdaptation(row)
        endTime = time.time()
        RandomcustomTime = endTime - startTime

        if RandomCustomAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], RandomCustomAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation Random:                 " + str(RandomCustomAdaptation[0:n_controllableFeatures]))
            print("Model confidence Random:                " + str(RandomCustomConfidence))
            print("Adaptation score Random:                " + str(RandomCustomScore) + " /" + str(100 * n_controllableFeatures))
        else:
            print("No adaptation found")
            RandomCustomScore = None

        print("Custom Random algorithm execution time: " + str(RandomcustomTime) + " s")
        print("-" * 100)

        externalFeatures = row[n_controllableFeatures:]

        startTime = time.time()
        nsga3Adaptation, nsga3Confidence, nsga3Score = nsga3Planner.findAdaptation(externalFeatures)
        endTime = time.time()
        nsga3Time = endTime - startTime

        print("Best NSGA3 adaptation:           " + str(nsga3Adaptation[:n_controllableFeatures]))
        print("Model confidence:                " + str(nsga3Confidence))
        print("Adaptation score:                " + str(nsga3Score) + " /" + str(100 * n_controllableFeatures))
        print("NSGA3 execution time:            " + str(nsga3Time) + " s")

        print("-" * 100)

        """
        SHAPscoreDiff = None
        FIscoreDiff = None
        FitestscoreDiff = None
        RandomScoreDiff = None
        SHAPscoreImprovement = None
        FIscoreImprovement = None
        FitestscoreImprovement = None
        RandomScoreImprovement = None

        
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

        if RandomcustomTime == 0:
            Randomspeedup = 0
        else:
            Randomspeedup = customTime / RandomcustomTime
        meanSpeedupRandom = (meanSpeedupRandom * (k - 1) + Randomspeedup) / k
        print(Fore.GREEN + "Speed-up Random: " + " " * 14 + str(Randomspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up Random: " + " " * 9 + str(meanSpeedupRandom) + "x")

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

        if FitestcustomTime == 0:
            Fitestspeedup = 0
        else:
            Fitestspeedup = customTime / FitestcustomTime
        meanSpeedupFitest = (meanSpeedupFitest * (k - 1) + Fitestspeedup) / k
        print(Fore.GREEN + "Speed-up Fitest: " + " " * 14 + str(Fitestspeedup) + "x")
        print(Style.RESET_ALL + Fore.YELLOW + "Mean speed-up Fitest: " + " " * 9 + str(meanSpeedupFitest) + "x")

        if FitestcustomAdaptation is not None and customAdaptation is not None:
            FitestscoreDiff = FitestcustomScore - customScore
            FitestscoreImprovement = FitestscoreDiff / customScore
            print("Score diff Fitest:        " + " " * 5 + str(FitestscoreDiff))
            print("Score improvement Fitest: " + " " * 5 + "{:.2%}".format(FitestscoreImprovement))
        else:
            failedAdaptationsFitest += 1

        if RandomcustomAdaptation is not None and customAdaptation is not None:
            RandomScoreDiff = RandomcustomScore - customScore
            RandomscoreImprovement = RandomScoreDiff / customScore
            print("Score diff Random:        " + " " * 5 + str(RandomScoreDiff))
            print("Score improvement Random: " + " " * 5 + "{:.2%}".format(RandomscoreImprovement))
        else:
            failedAdaptationsRandom += 1

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
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsFI) + customScore) / (
                    k - failedAdaptationsFI)
            meanScoreDiffFI = (meanScoreDiffFI * (k - 1 - failedAdaptationsFI) + FIscoreDiff) / (
                    k - failedAdaptationsFI)
            meanScoreImprovementFI = meanScoreDiffFI / meanCustomScore
            print("Mean score diff FI:        " + str(meanScoreDiffFI))
            print("Mean score improvement FI: " + "{:.2%}".format(meanScoreImprovementFI))
        
        if FitestcustomAdaptation is not None and customAdaptation is not None:
            meanCustomScoreFitest = (meanCustomScoreFitest * (k - 1 - failedAdaptationsFitest) + FitestcustomScore) / (
                    k - failedAdaptationsFitest)
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsFitest) + customScore) / (
                    k - failedAdaptationsFitest)
            meanScoreDiffFitest = (meanScoreDiffFitest * (k - 1 - failedAdaptationsFitest) + FitestscoreDiff) / (
                    k - failedAdaptationsFitest)
            meanScoreImprovementFitest = meanScoreDiffFitest / meanCustomScore
            print("Mean score diff Fitest:        " + str(meanScoreDiffFitest))
            print("Mean score improvement Fitest: " + "{:.2%}".format(meanScoreImprovementFitest))
        
        if RandomcustomAdaptation is not None and customAdaptation is not None:
            meanCustomScoreRandom = (meanCustomScoreRandom * (k - 1 - failedAdaptationsRandom) + RandomcustomScore) / (
                    k - failedAdaptationsRandom)
            meanCustomScore = (meanCustomScore * (k - 1 - failedAdaptationsRandom) + customScore) / (
                    k - failedAdaptationsRandom)
            meanScoreDiffRandom = (meanScoreDiffRandom * (k - 1 - failedAdaptationsRandom) + RandomScoreDiff) / (
                    k - failedAdaptationsRandom)
            meanScoreImprovementRandom = meanScoreDiffRandom / meanCustomScore
            print("Mean score diff Random:        " + str(meanScoreDiffRandom))
            print("Mean score improvement Random: " + "{:.2%}".format(meanScoreImprovementRandom))

        print(Style.RESET_ALL + "=" * 100)
        """

        results.append([customAdaptation,
                        customConfidence,
                        customScore,
                        customTime])

        resultsNSGA.append([nsga3Adaptation,
                            nsga3Confidence,
                            nsga3Score,
                            nsga3Time])

        resultsSHAP.append([SHAPcustomAdaptation,
                            SHAPcustomConfidence,
                            SHAPcustomScore,
                            SHAPcustomTime])

        resultsFI.append([FIcustomAdaptation,
                          FIcustomConfidence,
                          FIcustomScore,
                          FIcustomTime])

        resultsFitest.append([FitestcustomAdaptation,
                              FitestcustomConfidence,
                              FitestcustomScore,
                              FitestcustomTime])

        resultsRandom.append([RandomCustomAdaptation,
                              RandomCustomConfidence,
                              RandomCustomScore,
                              RandomcustomTime])

    results = pd.DataFrame(results, columns=["custom_adaptation",
                                             "custom_confidence",
                                             "custom_score",
                                             "custom_time"])

    resultsNSGA = pd.DataFrame(resultsNSGA, columns=["custom_adaptation",
                                                     "custom_confidence",
                                                     "custom_score",
                                                     "custom_time"])

    resultsSHAP = pd.DataFrame(resultsSHAP, columns=["custom_adaptation",
                                                     "custom_confidence",
                                                     "custom_score",
                                                     "custom_time"])

    resultsFI = pd.DataFrame(resultsFI, columns=["custom_adaptation",
                                                 "custom_confidence",
                                                 "custom_score",
                                                 "custom_time"])

    resultsFitest = pd.DataFrame(resultsFitest, columns=["custom_adaptation",
                                                         "custom_confidence",
                                                         "custom_score",
                                                         "custom_time"])

    resultsRandom = pd.DataFrame(resultsRandom, columns=["custom_adaptation",
                                                         "custom_confidence",
                                                         "custom_score",
                                                         "custom_time"])

    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)

    results.to_csv(path + "/results.csv")
    resultsSHAP.to_csv(path + "/resultsSHAP.csv")
    resultsFI.to_csv(path + "/resultsFI.csv")
    resultsFitest.to_csv(path + "/resultsFitest.csv")
    resultsRandom.to_csv(path + "/resultsRandom.csv")
    resultsNSGA.to_csv(path + "/resultsNSGA.csv")

    if evaluate:
        evaluateAdaptations(results, resultsSHAP, resultsFI, resultsFitest, resultsRandom, resultsNSGA, featureNames)

    programEndTime = time.time()
    totalExecutionTime = programEndTime - programStartTime
    print("\nProgram execution time: " + str(totalExecutionTime / 60) + " m")
