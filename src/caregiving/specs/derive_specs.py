"""Derive specs from the specs file."""

import numpy as np
import yaml


def read_and_derive_specs(spec_path):
    """Read and derive specs from the specs file."""
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1

    # Number of education types and choices from labels
    specs["n_education_types"] = len(specs["education_labels"])
    specs["n_sexes"] = len(specs["sex_labels"])
    specs["n_choices"] = len(
        specs["choice_labels"]
    )  # * len(specs["caregiving_labels"])

    # For health states, get number and var values for alive states
    specs["n_health_states"] = len(specs["health_labels"])
    specs["alive_health_vars"] = np.where(np.array(specs["health_labels"]) != "Death")[
        0
    ]
    specs["death_health_var"] = np.where(np.array(specs["health_labels"]) == "Death")[
        0
    ][0]

    if "health_labels_three" in specs.keys():
        specs["n_health_states_three"] = len(specs["health_labels_three"])
        specs["alive_health_vars_three"] = np.where(
            np.array(specs["health_labels_three"]) != "Death"
        )[0]
        specs["death_health_var_three"] = np.where(
            np.array(specs["health_labels_three"]) == "Death"
        )[0][0]

    # Partner states
    specs["n_partner_states"] = len(specs["partner_labels"])

    # you can retire from min retirement age until max retirement age
    specs["n_policy_states"] = (
        int(
            ((specs["max_SRA_baseline"] - specs["min_SRA"]) / specs["SRA_grid_size"])
            + 1
        )
        # + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA_baseline"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    return specs
