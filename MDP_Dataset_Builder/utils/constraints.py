# Utils for checking constraints

from mdp_simulator import *

"""
Example of constraint:

CONSTRAINTS = {
    "S0": {
        "a": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    },
    "S5": {
        "g": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    },
    "S10": {
        "l": [[0.0, 1.0], [0.0, 1.0]],
        "m": [[0.0, 1.0], [0.0, 1.0]],
    }
}
"""


def check_single_constraint_hdi(mdp: MDP, single_constraint: dict):
    states = mdp.get_states_dictionary()

    for state_id, state_constraints in single_constraint.items():
        state: SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action: Action = state.get_action(action_id)
            hdi_intervals = action.get_hdi()

            for hdi, bound in zip(hdi_intervals, bounds):
                if bound[0] > hdi[1] or bound[1] < hdi[0]:
                    return False
    return True


def check_single_constraint_expected(mdp: MDP, single_constraint: dict):
    states = mdp.get_states_dictionary()

    for state_id, state_constraints in single_constraint.items():
        state: SingleState = states.get(state_id)
        for action_id, bounds in state_constraints.items():
            action: Action = state.get_action(action_id)
            expected = action.get_expected()

            for expected_prob, bound in zip(expected, bounds):
                if bound[0] > expected_prob or bound[1] < expected_prob:
                    return False
    return True
