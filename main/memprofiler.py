import tracemalloc
import warnings
import numpy as np
import pandas as pd
from colorama import Fore, Style
from memory_profiler import memory_usage
import time
from sklearn.model_selection import train_test_split
from CustomPlanner import CustomPlanner
from SHAPCustomPlanner import SHAPCustomPlanner
from FICustomPlanner import FICustomPlanner
from FITEST import FitestPlanner
from NSGA3Planner import NSGA3Planner
from model.ModelConstructor import constructModel
from RandomCustomPlanner import RandomPlanner
from util import vecPredictProba


def profile_memory(func, *args, **kwargs):
    tracemalloc.start()
    func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(peak / (1024 ** 2))
    return peak / (1024 ** 2)

def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)


def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])


def main():
    warnings.filterwarnings("ignore")

    ds = pd.read_csv('../datasets/dataset5000.csv')
    featureNames = ['cruise speed','image resolution','illuminance','controls responsiveness','power','smoke intensity','obstacle size','obstacle distance','firm obstacle']
    controllableFeaturesNames = featureNames[0:4]
    externalFeaturesNames = featureNames[4:9]
    controllableFeatureIndices = [0, 1, 2, 3]

    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize
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

    controllableFeatureDomains = np.repeat([[0, 100]], n_controllableFeatures, axis=0)
    discreteIndices = []

    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                                  optimizationDirections, optimizationScore, 1, "../explainability_plots")

    nsga3Planner = NSGA3Planner(models, targetConfidence, controllableFeatureIndices, controllableFeatureDomains,
                                optimizationDirections, successScore, optimizationScore)

    SHAPcustomPlanner = SHAPCustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                          controllableFeaturesNames, controllableFeatureIndices,
                                          controllableFeatureDomains,
                                          optimizationDirections, optimizationScore, 1, "../explainability_plots")

    FIcustomPlanner = FICustomPlanner(X_train, y_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                      controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                                      optimizationDirections, optimizationScore, 1, "../explainability_plots")

    pop_size = nsga3Planner.algorithm.pop_size

    Fitest_planner = FitestPlanner(models, targetConfidence,
                                   controllableFeatureIndices, controllableFeatureDomains, optimizationScore,
                                   successScore, pop_size, discreteIndices, 4,
                                   [0.8, 0.8, 0.8, 0.8])

    Random_Planner = RandomPlanner(controllableFeatureIndices, controllableFeatureDomains, discreteIndices, models,
                                   optimizationScore)

    data_time = {
        'PDPTime': [customPlanner.PDPtime],
        'SPDPTime': [customPlanner.SPDPtime],
        'SHAPTime': [SHAPcustomPlanner.rankingTime],
        'FITime': [FIcustomPlanner.rankingTime]
    }

    data_memory = {
        'CustomMemory': [],
        'SHAPMemory': [],
        'FIMemory': [],
        'FitestMemory': [],
        'RandomMemory': [],
        'NSGA3Memory': [],
    }

    testNum = 200
    for k in range(1, testNum + 1):
        rowIndex = k - 1
        row = X_test.iloc[rowIndex, :].to_numpy()

        print(Fore.BLUE + "Test " + str(k) + ":" + Style.RESET_ALL)
        print("Row " + str(rowIndex) + ":\n" + str(row))
        print("-" * 100)


        customMemUsage = profile_memory(customPlanner.findAdaptation, row)
        SHAPcustomMemUsage = profile_memory(SHAPcustomPlanner.findAdaptation, row)
        FICustomMemUsage = profile_memory(FIcustomPlanner.findAdaptation, row)
        FitestMemUsage = profile_memory(Fitest_planner.run_search, row)
        RandomMemUsage = profile_memory(Random_Planner.findAdaptation, row)
        externalFeatures = row[n_controllableFeatures:]
        nsga3MemUsage = profile_memory(nsga3Planner.findAdaptation, externalFeatures)


        data_memory['CustomMemory'].append(customMemUsage)
        data_memory['SHAPMemory'].append(SHAPcustomMemUsage)
        data_memory['FIMemory'].append(FICustomMemUsage)
        data_memory['FitestMemory'].append(FitestMemUsage)
        data_memory['RandomMemory'].append(RandomMemUsage)
        data_memory['NSGA3Memory'].append(nsga3MemUsage)

        print("-" * 100)


    df_memory = pd.DataFrame(data_memory)
    df_time = pd.DataFrame(data_time)

    path = "../results/"


    df_memory.to_csv(path + "/memory_results.csv", index=False)
    df_time.to_csv(path + "/time_results.csv", index=False)
   #print(df_memory)
   #print(df_time)


if __name__ == '__main__':
    main()
