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


class Req:
    def __init__(self, name):
        self.name = name


def vecPredictProba(models, features):
    probas = []
    for model in models:
        probas.append(model.predict_proba([features])[0, 1])
    return probas


if __name__ == '__main__':
    os.chdir(sys.path[0])
    # suppress all warnings
    warnings.filterwarnings("ignore")

    ds = pd.read_csv('../datasets/dataset1000.csv')
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
    optimizationDirection = [1, -1, -1, -1]

    reqs = [Req("req_1")]#, Req("req_1"), Req("req_2"), Req("req_3")]
    for req in reqs:
        X = ds.loc[:, featureNames]
        y = ds.loc[:, req.name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        y_test = np.ravel(y_test)
        y_train = np.ravel(y_train)

        print(Fore.RED + "Requirement: " + req.name + "\n" + Style.RESET_ALL)

        req.model = constructModel(X_train.values, X_test.values, y_train, y_test)

        print("=======================================================================================================")

        # make pdps
        req.pdps = {}
        for i, f in enumerate(controllableFeaturesNames):
            path = '../plots/' + req.name + '/individuals'
            if not os.path.exists(path):
                os.makedirs(path)
            req.pdps[i] = pdp.partialDependencePlot(req.model, X_train, [f], "both", path + '/' + f + '.png')

        # create lime explainer
        req.limeExplainer = lime.createLimeExplainer(X_train)

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

    meanSpeedup = 0
    meanScoreDiff = 0
    meanScoreDiffPerc = 0
    meanScoreImprovement = 0
    meanScoreImprovementPerc = 0
    meanTimeAddition = 0

    # adaptations
    results = []
    deeperSearch = True

    # pdp max points of mean line can be computed a priori
    meanLineMaxPoints = {}
    for req in reqs:
        meanLineMaxPoints[req.name] = []
        for i in range(4):
            meanLineMaxPoints[req.name].append(pdp.getMaxPointOfMeanLine(req.pdps[i]))

    targetProbas = np.array([0.8, 0.8, 0.8, 0.8])

    variableMin = 0
    variableMax = 100
    variableDomainSize = variableMax - variableMin

    yDeltaMin = 1/100
    deltaMax = variableDomainSize/20

    for k in range(1, 20 + 1):
        random.seed()
        rowIndex = k - 1     # random.randrange(0, X.shape[0])
        row = X.iloc[rowIndex, :].to_numpy()
        print(Fore.BLUE + "Test " + str(k) + ":" + Style.RESET_ALL)
        print("Row " + str(rowIndex) + ":\n" + str(row))

        path = "../plots/adaptation/"
        if not os.path.exists(path):
            os.makedirs(path)

        for req in reqs:
            lime.saveExplanation(lime.explain(req.limeExplainer, req.model, row), path + req.name + "_starting")
        print("-------------------------------------------------------------------------------------------------------")

        models = []
        for req in reqs:
            models.append(req.model)

        startTime = time.time()

        # pdp max points of the closest line must be computed at each adaptation if needed
        closestLineMaxPoints = []
        for i in range(4):
            closestLineMaxPoints.append(pdp.getMaxPointOfClosestLine(summaryPdps[i], row[i], np.prod(vecPredictProba(models, features=row))))

        # trivial solution
        adaptation = np.copy(row)
        for i, best in enumerate(meanLineMaxPoints[reqs[0].name]):
            adaptation[i] = best

        lastAdaptation = np.copy(adaptation)

        step = 0
        excludedFeatures = []
        lastProbas = vecPredictProba(models, features=lastAdaptation)
        # check if the best possible solution doesn't solve the problem
        if (lastProbas > targetProbas).all():
            while (lastProbas > targetProbas).all() or len(excludedFeatures) < len(controllableFeaturesNames):
                # select the next feature to modify
                featureIndex = None
                bestIncrement = None
                bestSlope = None
                for i in range(len(controllableFeaturesNames)):
                    if i not in excludedFeatures:
                        slope = pdp.getSlopeOfClosestLine(summaryPdps[i], adaptation[i], np.prod(lastProbas))
                        increment = slope * optimizationDirection[i]
                        if bestIncrement is None or increment > bestIncrement:
                            featureIndex = i
                            bestSlope = slope
                            bestIncrement = increment

                # stop if no feature can be improved
                if featureIndex == -1:
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

                lastProbas = vecPredictProba(models, features=lastAdaptation)
                # print("proba: " + str(lastProbas) + "\n")

                if (lastProbas < targetProbas).all():
                    excludedFeatures.append(featureIndex)
                    lastAdaptation = np.copy(adaptation)
                    # print("discarded\n")
                else:
                    adaptation = np.copy(lastAdaptation)
                    # print("accepted\n")

                step += 1

        else:
            print("Adaptation not found, using best adaptation")
        endTime = time.time()
        customTime = endTime - startTime
        customAdaptation = np.copy(adaptation)

        print("Adaptation:\n" + str(customAdaptation[0:4]))
        custom_probas = vecPredictProba(models, features=customAdaptation)
        print("Model confidence:\t" + str(custom_probas))
        customAlgoScore = 400 - (100 - customAdaptation[0] + customAdaptation[1] + customAdaptation[2] + customAdaptation[3])
        print("Score of the adaptation:\t" + str(customAlgoScore) + "/400")
        print("Total steps:\t" + str(step))

        for req in reqs:
            lime.saveExplanation(lime.explain(req.limeExplainer, req.model, row), path + req.name + "_final")

        print("\nCustom algorithm execution time: " + str(customTime) + " s")
        print("-------------------------------------------------------------------------------------------------------")

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
        deeperAlgoTime = 0
        deeperAdaptation = None
        deeper_proba = None
        deeperAlgoScore = None
        deeperScoreImprovement = 0
        deeperScoreImprovementPerc = None

        constantFeatures = X.iloc[rowIndex, 4:9]
        startTime = time.time()
        res = nsga3(models, constantFeatures)
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
            nsga3Adaptation = res.X[bestAdaptationIndex][0]
            nsga3_probas = vecPredictProba(models, np.append(res.X[bestAdaptationIndex], constantFeatures))

            print("Best NSGA3 adaptation:")
            print(nsga3Adaptation)

            print("Model confidence: " + str(nsga3_probas))

            print("Score: " + str(nsga3Score))
        else:
            print("No adaptation found")
            nsga3Adaptation = None
            nsga3_probas = None
            nsga3Score = None

        print("\nNSGA3 execution time: " + str(nsga3Time) + " s")

        if res.X is not None:
            print("-------------------------------------------------------------------------------------------------------")
            if step > 0:
                speedup = nsga3Time / (customTime + deeperAlgoTime)
            else:
                speedup = None
            scoreDiff = customAlgoScore + deeperScoreImprovement - nsga3Score
            scoreDiffPercent = scoreDiff/nsga3Score
            scoreDiffPercentString = "{:.2%}".format(scoreDiffPercent)
            print(Fore.GREEN + "\nSpeed-up: " + str(speedup) + "x")
            print("Score diff: " + str(scoreDiff))
            print("Score diff [% of loss]: " + scoreDiffPercentString + Style.RESET_ALL)

            if step > 0:
                meanSpeedup = (meanSpeedup * (k - 1) + speedup) / k
            meanScoreDiff = (meanScoreDiff * (k - 1) + scoreDiff) / k
            meanScoreDiffPerc = (meanScoreDiffPerc * (k - 1) + scoreDiffPercent) / k
            print(Fore.YELLOW + "Mean speed-up: " + str(meanSpeedup) + "x")
            print("Mean score improvement with deeperAlgo: " + str(meanScoreImprovementPerc))
            print("Mean score diff: " + str(meanScoreDiff))
            print("Mean score diff [% of loss]: " + str("{:.2%}".format(meanScoreDiffPerc)) + "\n" + Style.RESET_ALL)
        else:
            scoreDiff = None
            scoreDiffPercentString = None
            speedup = None

        print("=======================================================================================================")

        results.append([nsga3Adaptation, customAdaptation, deeperAdaptation,
                        nsga3_probas, custom_probas, deeper_proba,
                        nsga3Score, customAlgoScore, deeperAlgoScore, deeperScoreImprovementPerc, scoreDiff, scoreDiffPercentString,
                        nsga3Time, customTime, deeperAlgoTime, speedup])

    results = pd.DataFrame(results, columns=["nsga3_adaptation", "custom_adaptation", "deeper_adaptation",
                                             "nsga3_confidence", "custom_confidence", "deeper_proba",
                                             "nsga3_score", "custom_score", "deeper_score", "deeper_improvement[%]", "score_diff", "score_diff[%]",
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
