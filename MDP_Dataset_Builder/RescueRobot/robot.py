from math import exp

import numpy as np

def step_function(x, step, speed):
    atan0 = np.arctan(-speed*step)
    atan1 = np.arctan(speed*(1-step))
    a = atan0 if step > 0 else atan1
    m = np.sign(speed) / (atan1 - atan0)
    return m * (np.arctan(speed * (x - step)) - a)

def segment(x, x1, x2, y1, y2):
    return (x - x1) / (x2 - x1) * (y2 - y1) + y1

def do_nothing(old_probability: np.ndarray, variable_value) -> np.ndarray:
    return old_probability

def power_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 15:
        increment = min(0.2, old_probability[1])

        old_probability[0] += increment / 2
        old_probability[2] += increment / 2
        old_probability[1] -= increment

    return old_probability

def cruise_speed_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.8)
        increment = old_probability[1] * (1 - multiplier) / 2

        old_probability[0] += increment
        old_probability[2] += increment
        old_probability[1] *= multiplier

    return old_probability

def illuminance_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0, 1)
        increment = old_probability[1] * (1 - multiplier) / 2

        old_probability[0] += increment
        old_probability[2] += increment
        old_probability[1] *= multiplier

    return old_probability

def smoke_intensity_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 30:
        multiplier = segment(variable_value, 30, 100, 1, 0)
        increment = old_probability[1] * (1 - multiplier) / 2

        old_probability[0] += increment
        old_probability[2] += increment
        old_probability[1] *= multiplier

    return old_probability

def obstacle_distance_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 80:
        multiplier = segment(variable_value, 80, 100, 1, 0.5)
        increment = old_probability[1] * (1 - multiplier) / 2

        old_probability[0] += increment
        old_probability[2] += increment
        old_probability[1] *= multiplier

    return old_probability

def firm_obstacle_s0_a(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 1:
        increment = min(0.1, old_probability[1])

        old_probability[0] += increment / 2
        old_probability[2] += increment / 2
        old_probability[1] -= increment

    return old_probability

def power_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 15:
        increment = min(0.2, old_probability[1])

        old_probability[0] += increment
        old_probability[1] -= increment

    return old_probability

def cruise_speed_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.8)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def illuminance_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0, 1)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def smoke_intensity_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 30:
        multiplier = segment(variable_value, 30, 100, 1, 0)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def obstacle_size_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 10:
        multiplier = segment(variable_value, 0, 10, 0.8, 1)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def obstacle_distance_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.5)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def firm_obstacle_s0_b(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment = min(0.15, old_probability[1])

        old_probability[0] += increment
        old_probability[1] -= increment

    return old_probability

def cruise_speed_s5_g(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 50:
        multiplier = segment(variable_value, 50, 100, 1, 0.7)
        increment = old_probability[2] * (1 - multiplier)

        old_probability[0] += increment / 3
        old_probability[1] += increment / 3 * 2
        old_probability[2] *= multiplier

    return old_probability

def obstacle_distance_s5_g(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0.1, 1)
        increment = old_probability[2] * (1 - multiplier)

        old_probability[0] += increment / 3
        old_probability[1] += increment / 3 * 2
        old_probability[2] *= multiplier

    return old_probability

def firm_obstacle_s5_g(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment = min(0.1, old_probability[2])

        old_probability[0] += increment / 3
        old_probability[1] += increment / 3 * 2
        old_probability[2] -= increment

    return old_probability

def cruise_speed_s6_h(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 50:
        multiplier = segment(variable_value, 50, 100, 1, 0.7)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def obstacle_distance_s6_h(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0.1, 1)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def power_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 15:
        increment = min(0.2, old_probability[0])

        old_probability[1] += increment
        old_probability[0] -= increment

    return old_probability

def cruise_speed_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.7)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def illuminance_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0, 1)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def smoke_intensity_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 30:
        multiplier = segment(variable_value, 30, 100, 1, 0)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def obstacle_size_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.8)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def obstacle_distance_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0.1, 1)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def firm_obstacle_s8_j(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment = min(0.1, old_probability[0])

        old_probability[1] += increment
        old_probability[0] -= increment

    return old_probability

def power_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 15:
        increment = min(0.2, old_probability[0])

        old_probability[1] += increment
        old_probability[0] -= increment

    return old_probability

def cruise_speed_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.7)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def illuminance_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0, 1)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def smoke_intensity_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 30:
        multiplier = segment(variable_value, 30, 100, 1, 0)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def obstacle_size_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.8)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def obstacle_distance_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0.1, 1)
        increment = old_probability[0] * (1 - multiplier)

        old_probability[1] += increment
        old_probability[0] *= multiplier

    return old_probability

def firm_obstacle_s10_l(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment = min(0.1, old_probability[0])

        old_probability[1] += increment
        old_probability[0] -= increment

    return old_probability

def cruise_speed_s10_m(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value >= 70:
        multiplier = segment(variable_value, 70, 100, 1, 0.5)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def obstacle_distance_s10_m(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value <= 30:
        multiplier = segment(variable_value, 0, 30, 0.1, 1)
        increment = old_probability[1] * (1 - multiplier)

        old_probability[0] += increment
        old_probability[1] *= multiplier

    return old_probability

def firm_obstacle_s10_m(old_probability: np.ndarray, variable_value) -> np.ndarray:
    if variable_value == 0:
        increment = min(0.1, old_probability[1])

        old_probability[0] += increment
        old_probability[1] -= increment

    return old_probability

SemanticSpaceVariable = [
    {
        "Name": "power",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 100],
        "Default": 100,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": power_s0_a
            },
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": power_s0_b
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": power_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": power_s10_l
            },
        ],
    },
    {
        "Name": "cruise speed",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 100],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": cruise_speed_s0_a
            },
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": cruise_speed_s0_b
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": cruise_speed_s5_g
            },
            {
                "StateId": "S6",
                "ActionId": "h",
                "Method": cruise_speed_s6_h
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": cruise_speed_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": cruise_speed_s10_l
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": cruise_speed_s10_m
            },
        ],
    },
    {
        "Name": "illuminance",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 100],
        "Default": 70.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": illuminance_s0_a
            },
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": illuminance_s0_b
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": illuminance_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": illuminance_s10_l
            },
        ],
    },
    {
        "Name": "smoke intensity",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 100],
        "Default": 0.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": smoke_intensity_s0_a
            },
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": smoke_intensity_s0_b
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": smoke_intensity_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": smoke_intensity_s10_l
            },
        ],
    },
    {
        "Name": "obstacle size",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 100],
        "Default": 50.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": obstacle_size_s0_b
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": obstacle_size_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": obstacle_size_s10_l
            },
        ],
    },
    {
        "Name": "obstacle distance",
        "Type": "SYS",
        "Domain": float,
        "Range": [0, 100],
        "Default": 20.0,
        "Combinations": [
            {
                "StateId": "S0",
                "ActionId": "a",
                "Method": obstacle_distance_s0_a
            },
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": obstacle_distance_s0_b
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": obstacle_distance_s5_g
            },
            {
                "StateId": "S6",
                "ActionId": "h",
                "Method": obstacle_distance_s6_h
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": obstacle_distance_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": obstacle_distance_s10_l
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": obstacle_distance_s10_m
            },
        ],
    },
    {
        "Name": "firm obstacle",
        "Type": "SYS",
        "Domain": int,
        "Range": [0, 1],
        "Default": 0,
        "Combinations": [{
                "StateId": "S0",
                "ActionId": "a",
                "Method": firm_obstacle_s0_a
            },
            {
                "StateId": "S0",
                "ActionId": "b",
                "Method": firm_obstacle_s0_b
            },
            {
                "StateId": "S5",
                "ActionId": "g",
                "Method": firm_obstacle_s5_g
            },
            {
                "StateId": "S8",
                "ActionId": "j",
                "Method": firm_obstacle_s8_j
            },
            {
                "StateId": "S10",
                "ActionId": "l",
                "Method": firm_obstacle_s10_l
            },
            {
                "StateId": "S10",
                "ActionId": "m",
                "Method": firm_obstacle_s10_m
            },
        ],
    },
]
