import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter
from util import vecPredictProba


class NSGA3Planner:
    def __init__(self, reqClassifiers, targetConfidence, successScoreFunction, optimizationScoreFunction):
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.successScoreFunction = successScoreFunction
        self.optimizationScoreFunction = optimizationScoreFunction

        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)

        # create the algorithm object
        self.algorithm = NSGA3(pop_size=455, ref_dirs=ref_dirs)

        self.termination = DefaultMultiObjectiveTermination(
            cvtol=1e-6,
            period=30,
            n_max_gen=1000
        )

        # create problem instance
        self.problem = Adaptation(reqClassifiers, targetConfidence, [])

    def findAdaptation(self, externalFeatures):
        # set problem
        self.problem.externalFeatures = externalFeatures

        # execute the optimization
        res = minimize(self.problem,
                       self.algorithm,
                       seed=1,
                       termination=self.termination)

        # Scatter().add(res.F).show()

        if res.X is not None:
            adaptations = res.X
        else:
            adaptations = np.array([individual.X for individual in res.pop])

        adaptations = np.append(adaptations, np.repeat([externalFeatures], adaptations.shape[0], axis=0), axis=1)
        optimizationScores = [self.optimizationScoreFunction(a) for a in adaptations]

        if res.X is not None:
            adaptationIndex = np.argmax(optimizationScores)
        else:
            successScores = [self.successScoreFunction(a, self.reqClassifiers, self.targetConfidence) for a in adaptations]
            adaptationIndex = np.argmax(successScores)

        adaptation = adaptations[adaptationIndex]
        confidence = vecPredictProba(self.reqClassifiers, [adaptation])[0]
        score = self.optimizationScoreFunction(adaptation)

        return adaptation, confidence, score


class Adaptation(Problem):
    def __init__(self, models, targetConfidence, externalFeatures):
        super().__init__(n_var=4, n_obj=4, n_constr=len(models), xl=0.0, xu=100.0)
        self.models = models
        self.targetConfidence = targetConfidence
        self.externalFeatures = externalFeatures

    def _evaluate(self, x, out, *args, **kwargs):
        xFull = np.c_[x, np.tile(self.externalFeatures, (x.shape[0], 1))]
        f1 = -x[:, 0]
        f2 = x[:, 1]
        f3 = x[:, 2]
        f4 = x[:, 3]

        out["F"] = [f1, f2, f3, f4]
        out["G"] = [self.targetConfidence[i] - self.models[i].predict_proba(xFull)[:, 1] for i in range(self.n_constr)]
