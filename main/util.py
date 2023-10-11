import os
import numpy as np


def vecPredictProba(models, X):
    if type(X) is list:
        X = np.array(X)

    probas = np.empty((X.shape[0], len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict_proba(X)[:, 1]
    probas = np.column_stack(np.split(probas, len(models)))
    return probas


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def evaluateAdaptations(dataset, name):
    os.chdir("../MDP_Dataset_Builder")
    np.save("./starting_combinations.npy", dataset)
    os.system("execute.bat ./starting_combinations.npy")
    os.system("merge_csvs.py")

    # Rename the file
    os.chdir("..")
    source_file = './MDP_Dataset_Builder/merge.csv'
    new_file = './results/' + name + '.csv'
    os.rename(source_file, new_file)
