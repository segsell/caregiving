import numpy as np
import yaml


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # Number of education types and choices from labels
    specs["n_education_types"] = len(specs["education_labels"])
    specs["n_sexes"] = len(specs["sex_labels"])
    specs["n_choices"] = len(specs["choice_labels"])

    # For health states, get number and var values for alive states
    specs["n_health_states"] = len(specs["health_labels"])
    specs["alive_health_vars"] = np.where(np.array(specs["health_labels"]) != "Death")[
        0
    ]
    specs["death_health_var"] = np.where(np.array(specs["health_labels"]) == "Death")[
        0
    ][0]

    # Partner states
    specs["n_partner_states"] = len(specs["partner_labels"])

    return specs
