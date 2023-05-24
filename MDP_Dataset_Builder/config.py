import argparse

MAX_SAMPLES = None  # 1000
INDEX_TO_RUN = None  # 0
TOTAL_TO_RUN = None  # 1

SS_VARIABLES = {
    "power": {"domain": int, "range": [0, 100]},
    "cruise speed": {"domain": float, "range": [0, 100]},
    "illuminance": {"domain": float, "range": [0, 100]},
    "smoke intensity": {"domain": float, "range": [0, 100]},
    "obstacle size": {"domain": float, "range": [0, 100]},
    "obstacle distance": {"domain": float, "range": [0, 100]},
    "firm obstacle": {"domain": int, "range": [0, 1]},
}

CONSTRAINT_EXAMPLE = {
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

low_missclassification = {
    "S0": {
        "a": [[0.0, 1.0], [0.0, 1.0], [0.0, 0.05]],
        "b": [[0.0, 0.05], [0.0, 1.0]]
    }
}

low_contact = {
    "S0": {
        "a": [[0.0, 0.03], [0.0, 1.0], [0.0, 1.0]]
    },
    "S5": {
        "g": [[0.0, 0.03], [0.0, 1.0], [0.0, 1.0]]
    },
    "S6": {
        "h": [[0.0, 0.03], [0.0, 1.0]]
    }
}

low_crash = {
    "S10": {
        "l": [[0.0, 1.0], [0.0, 0.03]],
        "m": [[0.0, 0.03], [0.0, 1.0]]
    }
}

safe = {
    "S5": {
        "g": [[0.0, 1.0], [0.0, 0.1], [0.0, 1.0]]
    },
    "S8": {
        "j": [[0.0, 1.0], [0.0, 0.1]]
    }
}

test = {
    "S6": {
        "h": [[0.0, 0.03], [0.0, 1.0]]
    }
}


CONSTRAINTS = [
    test,
    low_missclassification,
    low_contact,
    low_crash,
    safe
]

# Load arguments from cli

all_args = argparse.ArgumentParser()

all_args.add_argument("-m", "--max-samples", required=False, help="Max Samples to generate", type=int)
all_args.add_argument("-i", "--index-to-run", required=False, help="Index of this execution", type=int)
all_args.add_argument("-t", "--total-executions", required=False, help="Index of this execution", type=int)

args, _ = all_args.parse_known_args()
args = vars(args)

# Update default values

if args.get("max_samples") is not None:
    print(f"Max Samples: {args.get('max_samples')}")
    MAX_SAMPLES = args.get("max_samples")

if args.get("index_to_run") is not None:
    print(f"Index to run: {args.get('index_to_run')}")
    INDEX_TO_RUN = args.get("index_to_run")

if args.get("total_executions") is not None:
    print(f"Total executions: {args.get('total_executions')}")
    TOTAL_TO_RUN = args.get("total_executions")
