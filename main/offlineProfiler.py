import time
import explainability_techniques.PDP as pdp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from explainability_techniques.FeatureImportance import permutation_importance_classifier
from explainability_techniques.SHAP import shapClassifier
from model.ModelConstructor import constructModel
import sys
import os
import warnings
import tracemalloc
import csv
import subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def collect_time():

    pdps = {}
    start_time_pdp = time.time()
    for i, feature in enumerate(controllableFeaturesNames):
        for j, reqClassifier in enumerate(models):
            pdps[i] = []
            pdps[i].append(pdp.partialDependencePlot(reqClassifier, X_train, [feature], "both"))
    end_time_pdp = time.time()
    pdp_execution_time = end_time_pdp - start_time_pdp

    summaryPdps = []
    start_time_spdp = time.time()
    for i, feature in enumerate(controllableFeaturesNames):
        summaryPdps.append(pdp.multiplyPdps(pdps[i]))
    end_time_spdp = time.time()
    spdp_execution_time = end_time_spdp - start_time_spdp

    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataset_name, pdp_execution_time, spdp_execution_time])


    print(f"PDP Execution Time: {pdp_execution_time:.2f} seconds")
    print(f"SPDP Execution Time: {spdp_execution_time:.2f} seconds")

def collect_memory():

    tracemalloc.start()
    pdps = {}
    for i, feature in enumerate(controllableFeaturesNames):
        for j, reqClassifier in enumerate(models):
            pdps[i] = []
            pdps[i].append(pdp.partialDependencePlot(reqClassifier, X_train, [feature], "both"))

    current_memory, pdp_peak_memory = tracemalloc.get_traced_memory()
    pdp_peak_memory_mb = pdp_peak_memory / 1024 / 1024

    tracemalloc.reset_peak()

    summaryPdps = []

    for i, feature in enumerate(controllableFeaturesNames):
        summaryPdps.append(pdp.multiplyPdps(pdps[i]))

    current_memory, spdp_peak_memory = tracemalloc.get_traced_memory()
    spdp_peak_memory_mb = spdp_peak_memory / 1024 / 1024
    tracemalloc.stop()

    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataset_name, pdp_peak_memory_mb, spdp_peak_memory_mb])

    print(f"PDP Peak Memory Usage: {pdp_peak_memory_mb:.2f} MB")
    print(f"SPDP Peak Memory Usage: {spdp_peak_memory_mb:.2f} MB")


os.chdir(sys.path[0])
warnings.filterwarnings("ignore")
ds = pd.read_csv('../datasets/dataset5000.csv')
#featureNames = ['formation', 'flying_speed', 'countermeasure', 'weather', 'day_time', 'threat_range', '#threats']  #uav
featureNames = ['cruise speed', 'image resolution', 'illuminance', 'controls responsiveness', 'power',
                'smoke intensity', 'obstacle size', 'obstacle distance', 'firm obstacle']  #robot
#featureNames = ['car_speed','p_x','p_y', 'orientation','weather','road_shape'] #drive
controllableFeaturesNames = featureNames[0:4]
externalFeaturesNames = featureNames[4:9]
controllableFeatureIndices = [0, 1, 2, 3]

#reqs = ["req_0", "req_1", "req_2", "req_3", "req_4", "req_5", "req_6", "req_7", "req_8", "req_9", "req_10",
#       "req_11"]  #uav
reqs = ["req_0", "req_1", "req_2", "req_3"]  #robot
#reqs = ["req_0", "req_1", "req_2"] #drive

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

dataset_name = "DriveDouble"

csv_filename = "../results/profiling/time_pdp_spdp.csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "PDP_Time", "SPDP_Time"])
#for _ in range(20):
#    collect_time()

# Permutation Feature Importance Profiling
model_profile_data_FI = []

csv_filename = '../results/profiling/FI_profile.csv'

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Memory Peak (MB)'])

cumulative_importance = np.zeros(len(controllableFeatureIndices))
#tracemalloc.start()
for i, reqClassifier in enumerate(models):
    
    start_time = time.time()
    #tracemalloc.reset_peak()

    feature_indices = permutation_importance_classifier(reqClassifier, X_train, y_train, controllableFeatureIndices)
    end_time = time.time()
    elapsed_time = end_time - start_time
    #current_peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    model_name = type(reqClassifier).__name__
    print(model_name, elapsed_time)
    #model_profile_data_FI.append([elapsed_time])

#tracemalloc.stop()

#with open(csv_filename, mode='a', newline='') as file:
#    writer = csv.writer(file)
#    writer.writerows(model_profile_data_FI)


# SHAP Profiling
model_profile_data_SHAP = []
csv_filename = 'SHAP_profile.csv'

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Memory Peak (MB)'])

#tracemalloc.start()

#current_pid = os.getpid()
#subprocess.Popen(["powershell.exe", "./log_memory.ps1", "-ProcessID", str(current_pid), "-Interval", "60"])

for i, reqClassifier in enumerate(models):
    #tracemalloc.reset_peak()
    req = reqs[i]

    if not (isinstance(reqClassifier, LogisticRegression) or isinstance(reqClassifier, MLPClassifier)):
        start_time = time.time()
        feature_indices = shapClassifier(reqClassifier, X_train, controllableFeatureIndices)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #    current_peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        model_name = type(reqClassifier).__name__
        print(model_name, elapsed_time)
    #   model_profile_data_SHAP.append([model_name, current_peak])
    else:
        feature_indices = shapClassifier(reqClassifier, X_train, controllableFeatureIndices)
#tracemalloc.stop()

#with open(csv_filename, mode='a', newline='') as file:
#    writer = csv.writer(file)
#    writer.writerows(model_profile_data_SHAP)
