from utils.constraints_builder import compute_constraints

SS_VARIABLES = {
    "formation": { "domain": int, "range": [0, 1], },
    "flying_speed": { "domain": float, "range": [5., 50.], },
    "countermeasure": { "domain": int, "range": [0, 1], },
    "weather": { "domain": int, "range": [1, 4], },
    "day_time": { "domain": int, "range": [0, 23], },
    "threat_range": { "domain": float, "range": [1000.0, 40000.0], },
    "#threats": { "domain": int, "range": [1, 100], }
}

RQ = "rq1"
EXTRA_NAME = "NoHis"
PLOT = False

BATCH_SIZE = 5
HISTORY_RETRIES = 10
HISTORY_LEN = 1000000
MDP_FOLDER = "INPUT/UAV_v02"

MAX_STEPS = 100000

MAX_SAMPLES = 10000

# Constraints definition

IDEAL_SPOTS = {
    "S0": {
        "sTrt": [0.8, 0.05, 0.15]
    },
    "S10": {
        "sTrt": [0.44, 0.44, 0.02, 0.1]
    },
    "S20": {
        "sTrt": [0.95, 0.005, 0.045],
    }
}

constraints = compute_constraints([.035, .04, .045, .055], IDEAL_SPOTS)

CONSTRAINTS = [
    # SINGLE CONSTRAINTS
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        }
    },
    {
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        }
    },
    {
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0],
        }
    },
    # DUAL CONSTRAINTS
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        }
    },
    {
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0]
        }
    },
    {
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][3]
        }
    },
    # TRIPLE CONSTRAINTS
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][0]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0]
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][1]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][0],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1]
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][1],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][1]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][1],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][1],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][2]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][2],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][2],
        }
    },
    {
        "S0": {
            "sTrt": constraints["S0"]["sTrt"][3]
        },
        "S10": {
            "sTrt": constraints["S10"]["sTrt"][3],
        },
        "S20": {
            "sTrt": constraints["S20"]["sTrt"][3],
        }
    },
]

MINIMAL_CONSTRAINTS = {
    "S0": {
            "sTrt": constraints["S0"]["sTrt"][0]
    },
    "S10": {
        "sTrt": constraints["S10"]["sTrt"][0],
    },
    "S20": {
        "sTrt": constraints["S20"]["sTrt"][0],
    }
}

# _template = {
#     "S0": {
#         "sTrt": constraints["S0"]["sTrt"][0]
#     },
#     "S7": {
#         "sTrt": constraints["S7"]["sTrt"][0]
#     },
#     "S14": {
#         "sTrt": constraints["S14"]["sTrt"][0],
#         "sTrt": constraints["S14"]["sTrt"][0]
#     }
# }
