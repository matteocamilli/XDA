import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

class Adaptation(Problem):

    def __init__(self, model, constantFeatures, featureNames):
        super().__init__(n_var=4, n_obj=1, xl=0.0, xu=100.0)
        self.model = model
        self.constantFeatures = constantFeatures
        self.featureNames = featureNames

    def _evaluate(self, x, out, *args, **kwargs):
        xFull = np.c_[x, np.tile(self.constantFeatures, (x.shape[0], 1))]
        out["F"] = np.array(self.model.predict(pd.DataFrame(xFull, columns=self.featureNames)))
        # out["G"] = 0.1 - out["F"]

def nsga3(model, constantFeatures, featureNames):

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=12)

    # create the algorithm object
    algorithm = NSGA3(pop_size=92,
                      ref_dirs=ref_dirs)

    # execute the optimization
    res = minimize(Adaptation(model, constantFeatures, featureNames),
                   algorithm,
                   seed=1,
                   termination=('n_gen', 600))

    Scatter().add(res.F).show()

    return res