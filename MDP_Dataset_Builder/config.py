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

_tight_S0a_s5g = {
    "S0": {
        "a": [[0.0, 0.025], [0.945, 1.0], [0.0, 0.035]]
    },
    "S5": {
        "g": [[0.0, 0.025], [0.0, 0.035], [0.945, 1.0]]
    },
}

_medium_S0a_s5g = {
    "S0": {
        "a": [[0.0, 0.04], [0.93, 1.0], [0.0, 0.05]]
    },
    "S5": {
        "g": [[0.0, 0.04], [0.0, 0.05], [0.93, 1.0]]
    },
}

_loose_S0a_s5g = {
    "S0": {
        "a": [[0.0, 0.07], [0.9, 1.0], [0.0, 0.07]]
    },
    "S5": {
        "g": [[0.0, 0.07], [0.0, 0.07], [0.9, 1.0]]
    },
}

_tight_S10lm = {
    "S10": {
        "l": [[0.975, 1.0], [0.0, 0.025]],
        "m": [[0.0, 0.015], [0.985, 1.0]],
    }
}

_medium_S10lm = {
    "S10": {
        "l": [[0.955, 1.0], [0.0, 0.04]],
        "m": [[0.0, 0.03], [0.965, 1.0]],
    }
}

_loose_S10lm = {
    "S10": {
        "l": [[0.93, 1.0], [0.0, 0.08]],
        "m": [[0.0, 0.07], [0.93, 1.0]],
    }
}

_easypeasy = {
    "S0": {
        "a": [[0.0, 1.0], [0.0, 1.0], [0.0, 0.03]]
    }
}

CONSTRAINTS = [
    _tight_S0a_s5g,
    _medium_S0a_s5g,
    _loose_S0a_s5g,
    _tight_S10lm,
    _medium_S10lm,
    _loose_S10lm,
    _easypeasy
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
