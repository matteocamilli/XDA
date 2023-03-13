import mdp_simulator
from utils import build_sequences
from utils.file_proxy import dump_ss_variables, build_output_csv
from utils.constraints import *
import numpy as np


def __single_execution(ss_variables: dict, input_variables: np.ndarray) -> MDP:
    variables_to_exec = dict()
    for index, (key, value) in enumerate(ss_variables.items()):
        variables_to_exec.update({key: value['domain'](input_variables[index])})
    # dump_ss_variables(variables_to_exec)

    computed_mdp = mdp_simulator.run(override_ss_variables_starting_value=variables_to_exec)

    return computed_mdp


def run(ss_variables: dict, index: int, total: int, constraints: list[dict]):
    assert index < total

    constraints_satisfaction_output = []

    total_dataset = np.load("./starting_combinations.npy")
    total_elements = total_dataset.shape[0]

    starting_index = int((total_elements / total) * index)
    ending_index = int((total_elements / total) * (index + 1))

    partial_dataset = total_dataset.copy()
    partial_dataset = partial_dataset[starting_index:ending_index]

    for execution_index, input_variables in enumerate(partial_dataset):
        print(f"Execution n: {execution_index}")

        result_sat = []

        result_mdp = __single_execution(ss_variables, input_variables)

        for constraint in constraints:
            local_sat = check_single_constraint_hdi(result_mdp, constraint)
            result_sat.append(local_sat)

        constraints_satisfaction_output.append(result_sat)
    constraints_satisfaction_output = np.array(constraints_satisfaction_output)

    return {"input_data": partial_dataset, "output_data": constraints_satisfaction_output}


if __name__ == "__main__":
    print("HI")
