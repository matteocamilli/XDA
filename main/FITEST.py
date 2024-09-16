import numpy as np
from candidates import Candidate
from util import vecPredictProba
import copy
import random


class FitestPlanner:
    def __init__(self, reqClassifiers, targetConfidence, controllableFeatureIndices, controllableFeaturesDomains,
                 optimizationFunctionScore, successScoreFunction, pop_size, discreteIndices, no_of_Objectives,
                 threshold_criteria):
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.controllableFeatureIndices = np.array(controllableFeatureIndices)
        self.controllableFeaturesDomains = controllableFeaturesDomains
        self.optimizationFunctionScore = optimizationFunctionScore
        self.successScoreFunction = successScoreFunction
        self.pop_size = pop_size
        self.targetProba = 0.8
        self.discreteIndices = discreteIndices
        self.no_of_Objectives = no_of_Objectives
        self.threshold_criteria = threshold_criteria
        self.objective_uncovered = list(range(no_of_Objectives))

    def environment_selection(self, objective_uncovered, R_T):
        to_return = []
        for obj in objective_uncovered:
            min = 10000
            selected_candidate = None
            for candidate in R_T:
                if candidate.get_objective_value(obj) < min:
                    min = candidate.get_objective_value(obj)
                    selected_candidate = candidate
            if selected_candidate not in to_return:
                if selected_candidate is not None:
                    to_return.append(selected_candidate)

        return to_return

    def calculate_minimum_distance(self, candidate, random_pop):
        distance = 1000
        for each_candidate in random_pop:
            vals = each_candidate.get_candidate_values()
            candidate_vals = candidate.get_candidate_values()
            dist = np.linalg.norm(np.array(vals) - np.array(candidate_vals))
            if dist < distance:
                distance = dist
        return distance

    def generate_adaptive_random_population(self, size, lb, ub, discreteIndices, base_vector, controllable_indices,
                                            i=0):
        rp = self.generate_random_population(size, lb, ub, discreteIndices, base_vector)

        return rp

    # random value generator
    def generate_random_population(self, size, lb, ub, discreteIndices, base_vector):
        random_pop = []

        for _ in range(size):
            candidate_vals = base_vector.copy().tolist()

            for index in self.controllableFeatureIndices:
                if index in discreteIndices:
                    candidate_vals[index] = int(np.random.uniform(lb[index], ub[index]))
                else:
                    candidate_vals[index] = np.random.uniform(lb[index], ub[index])

            random_pop.append(Candidate(candidate_vals))

        return random_pop

    # dominates method, same from paper
    def dominates(self, value_from_pop, value_from_archive, objective_uncovered):
        dominates_f1 = False
        dominates_f2 = False
        for each_objective in objective_uncovered:
            f1 = value_from_pop[each_objective]
            f2 = value_from_archive[each_objective]
            if f1 < f2:
                dominates_f1 = True
            if f2 < f1:
                dominates_f2 = True
            if dominates_f1 and dominates_f2:
                break
        if dominates_f1 == dominates_f2:
            return False
        elif dominates_f1:
            return True
        return False

    def evaulate_population(self, func, pop, models):
        for candidate in pop:
            if isinstance(candidate, Candidate):
                result = func(models, [candidate.get_candidate_values()])
                candidate.set_objective_values(result)

    def exists_in_archive(self, archive, index):
        for candidate in archive:
            if candidate.exists_in_satisfied(index):
                return True
        return False

    # searching archive
    def get_from_archive(self, obj_index, archive):
        for candIndx in range(len(archive)):
            candidate = archive[candIndx]
            if candidate.exists_in_satisfied(obj_index):
                return candidate, candIndx
        return None

    # updating archive with adding the number of objective it satisfies [1,2,3,4,[ob1,ob2, objective index]]
    def update_archive(self, pop, objective_uncovered, archive, no_of_Objectives, threshold_criteria):
        for objective_index in range(no_of_Objectives):
            for pop_index in range(len(pop)):
                objective_values = pop[pop_index].get_objective_values()
                if objective_values[objective_index] >= threshold_criteria[objective_index]:
                    if self.exists_in_archive(archive, objective_index):
                        archive_value, cand_indx = self.get_from_archive(objective_index, archive)
                        obj_archive_values = archive_value.get_objective_values()
                        if obj_archive_values[objective_index] > objective_values[objective_index]:
                            value_to_add = pop[pop_index]
                            value_to_add.add_objectives_covered(objective_index)
                            # archive.append(value_to_add)
                            archive[cand_indx] = value_to_add
                            if objective_index in objective_uncovered:
                                objective_uncovered.remove(objective_index)
                            # archive.remove(archive_value)
                    else:
                        value_to_add = pop[pop_index]
                        value_to_add.add_objectives_covered(objective_index)
                        archive.append(value_to_add)
                        if objective_index in objective_uncovered:
                            objective_uncovered.remove(objective_index)
                else:
                    value_to_add = pop[pop_index]
                    value_to_add.add_objectives_covered(objective_index)
                    archive.append(value_to_add)

    # method to get the most dominating one
    def select_best(self, tournament_candidates, objective_uncovered):
        best = tournament_candidates[0]  # in case none is dominating other
        for i in range(len(tournament_candidates)):
            candidate1 = tournament_candidates[i]
            for j in range(len(tournament_candidates)):
                candidate2 = tournament_candidates[j]
                if (
                        self.dominates(candidate1.get_objective_values(), candidate2.get_objective_values(),
                                       objective_uncovered)):
                    best = candidate1
        return best

    def tournament_selection_improved(self, pop, size, objective_uncovered):
        tournament_candidates = []
        for i in range(size):
            indx = random.randint(0, len(pop) - 1)
            random_candidate = pop[indx]
            tournament_candidates.append(random_candidate)

        best = self.select_best(tournament_candidates, objective_uncovered)
        return best;

    def tournament_selection(self, pop, size, objective_uncovered):
        tournament_candidates = []
        for i in range(size):
            indx = random.randint(0, len(pop) - 1)
            random_candidate = pop[indx]
            tournament_candidates.append(random_candidate)

        best = self.select_best(tournament_candidates, objective_uncovered)
        return best;

    def do_single_point_crossover(self, parent1, parent2, controllable_indices):
        parent1 = parent1.get_candidate_values()
        parent2 = parent2.get_candidate_values()

        t_parent1 = parent1.copy()
        t_parent2 = parent2.copy()

        # Seleziona un punto di crossover solo tra le controllable features
        crossover_index = random.randint(1, len(controllable_indices) - 1)
        crossover_point = controllable_indices[crossover_index]

        # Scambia le controllable features dopo il punto di crossover
        for i in range(crossover_index, len(controllable_indices)):
            index = controllable_indices[i]
            t_parent1[index], t_parent2[index] = t_parent2[index], t_parent1[index]

        return Candidate(t_parent1), Candidate(t_parent2)

    def do_uniform_mutation(self, parent1, parent2, lb, ub, threshold):
        child1 = []
        child2 = []

        parent1 = parent1.get_candidate_values();
        parent2 = parent2.get_candidate_values()

        for parent1_index in range(len(self.controllableFeatureIndices)):
            probability_mutation = random.uniform(0, 1)
            if probability_mutation <= threshold:
                random_value = random.uniform(lb[parent1_index], ub[parent1_index])
                if parent1_index in self.discreteIndices:
                    child1.append(int(random_value))
                else:
                    child1.append(random_value)
            else:
                child1.append(parent1[parent1_index])

        for parent2_index in range(len(self.controllableFeatureIndices)):
            probability_mutation = random.uniform(0, 1)
            if probability_mutation <= threshold:  # 1/4         25% probability
                random_value = random.uniform(lb[parent2_index], ub[parent2_index])
                if parent2_index in self.discreteIndices:
                    child2.append(int(random_value))
                else:
                    child2.append(random_value)
            else:
                child2.append(parent2[parent2_index])

        return Candidate(child1), Candidate(child2)

    def correct(self, Q_T, lb, ub, integerFeatureIndices):
        for indx in range(len(Q_T)):
            candidate = Q_T[indx]
            values = candidate.get_candidate_values()
            for value_index in range(len(lb)):
                current_value = values[value_index]

                Q_T[indx].set_candidate_values_at_index(value_index, current_value)

                if current_value > ub[value_index] or current_value < lb[value_index]:
                    temp = \
                        self.generate_random_population(1, lb, ub, integerFeatureIndices, np.array(values),
                                                        )[
                            0]
                    new_value = temp.get_candidate_values()[value_index]

                    if integerFeatureIndices and value_index in integerFeatureIndices:
                        new_value = round(new_value)

                    Q_T[indx].set_candidate_values_at_index(value_index, new_value)

        return Q_T

    def do_simulated_binary_crossover(self, parent1, parent2, nc=20):
        parent1 = parent1.get_candidate_values();
        parent2 = parent2.get_candidate_values()
        u = random.uniform(0, 1)
        # half Raja's code, as the child candidates was too close
        if u < 0.5:
            B = (2 * u) ** (1 / (nc + 1))
        else:
            B = (1 / (2 * (1 - u))) ** (1 / (nc + 1))
        t_parent1 = []
        t_parent2 = []

        for indx in range(len(parent1)):
            if indx in self.controllableFeatureIndices:
                x1 = parent1[indx]
                x2 = parent2[indx]
                x1new = 0.5 * (((1 + B) * x1) + ((1 - B) * x2))
                x2new = 0.5 * (((1 - B) * x1) + ((1 + B) * x2))
                t_parent1.append(x1new)
                t_parent2.append(x2new)
            else:
                t_parent1.append(parent1[indx])
                t_parent2.append(parent2[indx])

        return Candidate(t_parent1), Candidate(t_parent2)

    def do_gaussain_mutation_for_one(self, parent1_cand, lb, ub, thresh):
        parent1 = parent1_cand.get_candidate_values()

        for attrib in range(len(lb)):
            if random.uniform(0, 1) > thresh:
                continue
            mu = 0;
            sigma = 1
            alpha = np.random.normal(mu, sigma)
            actualValueP1 = parent1[attrib];

            if (alpha < 1) and (alpha >= 0):
                if actualValueP1 + 1 < ub[attrib]:
                    parent1[attrib] = parent1[attrib] + 1;
            elif (alpha <= 0) and (alpha > -1):
                if actualValueP1 - 1 > lb[attrib]:
                    parent1[attrib] = parent1[attrib] - 1;
            else:
                if actualValueP1 + alpha < ub[attrib]:
                    parent1[attrib] = parent1[attrib] + alpha;
        return Candidate(parent1)

    def do_gaussain_mutation(self, parent1_cand, parent2_cand, lb, ub, thresh):
        parent1 = parent1_cand.get_candidate_values()
        parent2 = parent2_cand.get_candidate_values()
        for attrib in range(len(lb)):
            random_value_for_theshold = random.uniform(0, 1);
            if random_value_for_theshold > thresh:
                continue
            mu = 0;
            sigma = 1

            alpha = np.random.normal(mu, sigma)
            actualValueP1 = parent1[attrib];
            actualValueP2 = parent2[attrib];

            if (alpha < 1) and (alpha >= 0):
                if actualValueP1 + 1 < ub[attrib]:
                    parent1[attrib] = parent1[attrib] + 1;
                if actualValueP2 + 1 < ub[attrib]:
                    parent2[attrib] = parent2[attrib] + 1;

            elif (alpha <= 0) and (alpha > -1):
                if actualValueP1 - 1 > lb[attrib]:
                    parent1[attrib] = parent1[attrib] - 1;
                if actualValueP2 - 1 > lb[attrib]:
                    parent2[attrib] = parent2[attrib] - 1;
            else:
                if actualValueP1 + alpha < ub[attrib]:
                    parent1[attrib] = parent1[attrib] + alpha;
                if actualValueP2 + alpha < ub[attrib]:
                    parent2[attrib] = parent2[attrib] + alpha

        return Candidate(parent1), Candidate(parent2)

    def get_distribution_index(self, parent1, parent2, objective_uncovered, threshold_criteria):
        total = 0;
        for each_obj in objective_uncovered:
            total = total + parent1.get_objective_value(each_obj) - 0.95
            total = total + parent2.get_objective_value(each_obj) - 0.95

        total = total / (len(objective_uncovered) * 2)

        return 21 - (total * 400)

    def recombine_improved(self, pop, objective_uncovered, lb, ub, threshold_criteria):
        size = len(objective_uncovered)

        population_to_return = []

        if size == 1:
            candidate = self.do_gaussain_mutation_for_one(pop[0], lb, ub, (1 / len(pop[0].get_candidate_values())))
            population_to_return.append(candidate)

        else:
            while len(population_to_return) < size:
                parent1 = self.tournament_selection_improved(pop, 2,
                                                             objective_uncovered)  # tournament selection same size as paper
                parent2 = self.tournament_selection_improved(pop, 2, objective_uncovered)
                while parent1 == parent2:
                    parent2 = self.tournament_selection_improved(pop, 2, objective_uncovered)
                probability_crossover = random.uniform(0, 1)
                if probability_crossover <= 0.60:  # 60% probability
                    print("getting distribution index")
                    nc = self.get_distribution_index(parent1, parent2, objective_uncovered, threshold_criteria);
                    parent1, parent2 = self.do_simulated_binary_crossover(parent1, parent2, nc)
                child1, child2 = self.do_gaussain_mutation(parent1, parent2, lb, ub,
                                                           (1 / len(parent1.get_candidate_values())))

                population_to_return.append(child1)
                population_to_return.append(child2)

        return population_to_return
        # 0 Road type [categorical]
        # 1 Road ID [categorical]
        # 2 Scenario Length [categorical]
        # 3 Vehicle_in_front [categorical]
        # 4 vehicle_in_adjcent_lane [categorical]
        # 5 vehicle_in_opposite_lane [categorical]
        # 6 vehicle_in_front_two_wheeled [categorical]
        # 7 vehicle_in_adjacent_two_wheeled [categorical]
        # 8 vehicle_in_opposite_two_wheeled [categorical]
        # 9 time of day [ordinal]
        # 10 weather [ordinal]
        # 11 Number of People [categorical]
        # 12 Target Speed [numeric]
        # 13 Trees in scenario [categorical]
        # 14 Buildings in Scenario [categorical]
        # 15 task [categorical]

    # def calculate_distance(list_x, list_y):
    #     distance = 0
    #     for i in range(len(list_x)):
    #         if 0 <= i < 9:
    #             if list_x[i] != list_y[i]:
    #                     distance = distance+1
    #         elif 9 <= i < 11:
    #             if i == 9:
    #                 d1 = list_x[i]/2
    #                 d2 = list_y[i]/2
    #                 distance = distance + abs(d1-d2)
    #             if i == 10:
    #                 d1 = list_x[i]/6
    #                 d2 = list_y[i]/6
    #                 distance = distance + abs(d1-d2)
    #         elif i == 11:
    #             if list_x[i] != list_y[i]:
    #                     distance = distance+1
    #         elif i == 12:
    #             d1 = list_x[i]-2 / 2
    #             d2 = list_y[i]-2 / 2
    #             distance = distance + abs(d1 - d2)
    #         else:
    #             if list_x[i] != list_y[i]:
    #                 distance = distance + 1
    #     return distance

    def recombine(self, pop, objective_uncovered, lb, ub):
        size = len(pop)

        population_to_return = []
        if size == 1:
            candidate = self.do_gaussain_mutation_for_one(pop[0], lb, ub, (1 / len(pop[0].get_candidate_values())))
            population_to_return.append(candidate)
        else:
            while len(population_to_return) < size:
                parent1 = self.tournament_selection(pop, 2, objective_uncovered)  # tournament selection same size as paper
                parent2 = self.tournament_selection(pop, 2, objective_uncovered)
                while parent1 == parent2:
                    # print(len(pop))
                    # print(pop[0].get_candidate_values())
                    # print(pop[1].get_candidate_values())
                    parent2 = self.tournament_selection(pop, 2, objective_uncovered)
                probability_crossover = random.uniform(0, 1)
                if probability_crossover <= 0.60:  # 60% probability
                    parent1, parent2 = self.do_simulated_binary_crossover(parent1, parent2)
                child1, child2 = self.do_gaussain_mutation(parent1, parent2, lb, ub,
                                                           (1 / len(parent1.get_candidate_values())))

                population_to_return.append(child1)
                if len(population_to_return) < size:
                    population_to_return.append(child2)

        return population_to_return

    def run_search(self, base_vector):
        threshold_criteria = self.threshold_criteria
        objective_uncovered = []
        archive = []
        lb = self.controllableFeaturesDomains[:, 0]
        ub = self.controllableFeaturesDomains[:, 1]
        for obj in range(self.no_of_Objectives):
            objective_uncovered.append(obj)  # initialising number of uncovered objective

        random_population = self.generate_adaptive_random_population(self.pop_size, lb, ub, self.discreteIndices,
                                                                     base_vector,
                                                                     self.controllableFeatureIndices)  # Generating random population

        P_T = copy.copy(random_population)

        self.evaulate_population(vecPredictProba, random_population,
                                 self.reqClassifiers)  # evaluating whole generation and storing results

        self.update_archive(random_population, objective_uncovered, archive, self.no_of_Objectives,
                            threshold_criteria)  # updateing archive

        iteration = 0
        while iteration < 1000:
            iteration = iteration + 1  # iteration count
            #        logger.info("Uncov objs: "+str(objective_uncovered))

            R_T = []

            # print("Starting recombine")
            Q_T = self.recombine(P_T, objective_uncovered,
                                 lb, ub)  #
            # print("Recombine")
            Q_T = self.correct(Q_T, lb, ub, self.discreteIndices)
            # print("correct")
            # print(Q_T)

            #        sys.exit(0)

            self.evaulate_population(vecPredictProba, Q_T, self.reqClassifiers)  # evaluating offspring
            # print("evaulate_population(func, Q_T) ")
            self.update_archive(Q_T, objective_uncovered, archive, self.no_of_Objectives,
                                threshold_criteria)  # updating archive

            # print("Update ARchive")
            R_T = copy.deepcopy(P_T)  # R_T = P_T union Q_T
            R_T.extend(Q_T)
            if len(objective_uncovered) == 0:  # checking if all objectives are covered
                break
            P_T_1 = self.environment_selection(objective_uncovered, R_T)
            #  print("Environment selections")
            P_T = P_T_1  # assigning PT+1 to PT

        #  print(len(P_T))

        def solutionRank(c):
            return np.sum(c - self.targetConfidence)

        adaptationsRanks = [solutionRank(archive[i].get_objective_values()) for i in range(len(archive))]

        bestAdaptationIndice = np.where(adaptationsRanks == np.max(adaptationsRanks))[0][0]
        bestAdaptation = np.array(archive[bestAdaptationIndice].get_candidate_values())
        bestConfidence = archive[bestAdaptationIndice].get_objective_values()
        bestScore = self.optimizationFunctionScore(bestAdaptation)
        return bestAdaptation, bestConfidence, bestScore
