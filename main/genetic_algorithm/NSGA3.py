import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter

class Adaptation(Problem):

    def __init__(self, model, constantFeatures, featureNames):
        super().__init__(n_var=4, n_obj=4, n_constr=1, xl=0.0, xu=100.0)
        self.model = model
        self.constantFeatures = constantFeatures
        self.featureNames = featureNames

    def _evaluate(self, x, out, *args, **kwargs):
        xFull = np.c_[x, np.tile(self.constantFeatures, (x.shape[0], 1))]
        f1 = -x[:, 0]
        f2 = x[:, 1]
        f3 = x[:, 2]
        f4 = x[:, 3]

        out["F"] = [f1, f2, f3, f4]
        out["G"] = 0.8 - self.model.predict_proba(pd.DataFrame(xFull, columns=self.featureNames))[:, 1]

def nsga3(model, constantFeatures, featureNames):

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)

    # create the algorithm object
    algorithm = NSGA3(pop_size=455,
                      ref_dirs=ref_dirs)

    termination = DefaultMultiObjectiveTermination(
        cvtol=1e-6,
        period=30,
        n_max_gen=1000
    )

    # execute the optimization
    res = minimize(Adaptation(model, constantFeatures, featureNames),
                   algorithm,
                   seed=1,
                   termination=termination)

    # Scatter().add(res.F).show()

    return res