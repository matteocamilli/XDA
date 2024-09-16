import cProfile
import pstats
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from FICustomPlanner import FICustomPlanner
from model.ModelConstructor import constructModel


def optimizationScore(adaptation):
    return 400 - (100 - adaptation[0] + adaptation[1] + adaptation[2] + adaptation[3])

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
    models.append(constructModel(X_train.values,
                                 X_test.values,
                                 np.ravel(y_train.loc[:, req]),
                                 np.ravel(y_test.loc[:, req])))

controllableFeatureDomains = np.repeat([[0, 100]], n_controllableFeatures, axis=0)

planner = FICustomPlanner(X_train, y_train, n_neighbors, n_startingSolutions, models,
                          targetConfidence,
                          controllableFeaturesNames, [0, 1, 2, 3], controllableFeatureDomains,
                          optimizationDirections, optimizationScore, 1, "../explainability_plots")

pr = cProfile.Profile()
pr.enable()
testNum = 200
for k in range(1, testNum + 1):
    rowIndex = k - 1
    row = X_test.iloc[rowIndex, :].to_numpy()
    planner.findAdaptation(row)

pr.disable()

s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
