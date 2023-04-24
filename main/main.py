import pandas as pd

from data_preprocessing.Collinearity import collCheck, multiCheck
from data_preprocessing.Rebalancing import overSampling, underSampling, syntheticOverSampling

if __name__ == '__main__':

    #Test.test()
    ds = pd.read_csv('datasets/data1.csv')
    features = ["power",
                "cruise speed",
                "bandwidth",
                "quality",
                "illuminance",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]
    outcomes = ["req_0"]
                #, "req_1", "req_2", "req_3", "req_4", "req_5"
    X = ds.loc[:, features]
    y = ds.loc[:, outcomes]
    collCheck(X)
    multiCheck(X)
    overSampling(X, y)
    underSampling(X, y)
    syntheticOverSampling(X, y)