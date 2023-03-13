# Set of functions to work with files
import numpy as np
import pandas as pd

DEFAULT_SS_VARIABLES = {
    "power": 0,
    "cruise speed": 0.0,
    "bandwidth": 10.0,
    "quality": 0,
    "illuminance": 40.0,
    "smoke intensity": 0,
    "obstacle size": 0.0,
    "obstacle distance": 0.0,
    "firm obstacle": 0,
}


def dump_ss_variables(ss_variables, filename="./RescueRobot/ss_variables.mdp"):
    with open(filename, "w") as file:
        file.write("SEMANTIC_SPACE_VARIABLES")
        for key, value in ss_variables.items():
            file.write(f"\n{key}: {value};")


def dump_sets_to_csv(sets, filename="./starting_combinations.npy"):
    sets = list(map(lambda x: list(x), sets))

    sets_np = np.array(sets)

    np.save(filename, sets_np)


def build_output_csv(ss_variables, input_data, output_data, filename):
    dataframe = dict()

    for index, key in enumerate(ss_variables.keys()):
        dataframe.update({key: input_data[:, index]})

    a = pd.DataFrame(dataframe)

    # bool_list = np.random.choice([True, False], size=100 * 2).reshape(-1, 2)

    dataframe2 = dict()

    for index in range(output_data.shape[-1]):
        dataframe2.update({f"req_{index}": output_data[:, index]})

    b = pd.DataFrame(dataframe2)

    c = pd.concat([a, b], axis=1, join='inner')

    c.to_csv(filename, index=False)


if __name__ == "__main__":
    dump_ss_variables(DEFAULT_SS_VARIABLES)
