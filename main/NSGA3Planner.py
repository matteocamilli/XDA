import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from util import vecPredictProba


class NSGA3Planner:
    def __init__(self, reqClassifiers, targetConfidence, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections, successScoreFunction, optimizationScoreFunction):
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.successScoreFunction = successScoreFunction
        self.optimizationScoreFunction = optimizationScoreFunction

        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", len(controllableFeatureIndices), n_partitions=12)

        # create the algorithm object
        self.algorithm = NSGA3(ref_dirs=ref_dirs)

        self.termination = DefaultMultiObjectiveTermination(
            cvtol=1e-6,
            period=30,
            n_max_gen=1000
        )

        # create problem instance
        self.problem = Adaptation(reqClassifiers, targetConfidence, self.algorithm.pop_size, controllableFeatureIndices,
                                  controllableFeatureDomains, optimizationDirections)

    def findAdaptation(self, externalFeatures):
        # set problem
        self.problem.externalFeatures = externalFeatures

        # execute the optimization
        res = minimize(self.problem,
                       self.algorithm,
                       seed=1,
                       termination=self.termination)

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
    @property
    def externalFeatures(self):
        return self._externalFeatures

    @externalFeatures.setter
    def externalFeatures(self, externalFeatures):
        self._externalFeatures = np.repeat([externalFeatures], self.popSize, axis=0)

    def __init__(self, models, targetConfidence, popSize, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections):
        super().__init__(n_var=len(controllableFeatureIndices), n_obj=len(controllableFeatureIndices),
                         n_constr=len(models), xl=controllableFeatureDomains[:, 0], xu=controllableFeatureDomains[:, 1])
        self.models = models
        self.targetConfidence = np.repeat([targetConfidence], popSize, axis=0)
        self.controllableFeatureIndices = controllableFeatureIndices
        self.optimizationDirections = optimizationDirections
        self.popSize = popSize
        self.externalFeatures = []

    def _evaluate(self, x, out, *args, **kwargs):
        xFull = np.empty((self.popSize, self.n_var + self.externalFeatures.shape[1]))
        xFull[:, self.controllableFeatureIndices] = x
        externalFeatureIndices = np.delete(np.arange(xFull.shape[1]), self.controllableFeatureIndices)
        xFull[:, externalFeatureIndices] = self.externalFeatures

        out["F"] = [-self.optimizationDirections[i] * x[:, i] for i in range(x.shape[1])]
        out["G"] = [self.targetConfidence - vecPredictProba(self.models, xFull)]
