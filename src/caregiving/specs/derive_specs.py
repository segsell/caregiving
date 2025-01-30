import pickle as pkl

import jax.numpy as jnp
import numpy as np
import yaml

from caregiving.specs.experience_specs import create_max_experience
from caregiving.specs.family_specs import (
    predict_children_by_state,
    read_in_partner_transition_specs,
)
from caregiving.specs.income_specs import add_income_specs

# def generate_derived_and_data_derived_specs(path_dict, load_precomputed=False):
#     """This function reads in specs and adds derived and data estimated specs."""
#     specs = read_and_derive_specs(path_dict["specs"])

#     # Add income specs
#     specs = add_income_specs(specs, path_dict)

#     # family transitions
#     specs["children_by_state"] = predict_children_by_state(path_dict, specs)

#     # Read in family transitions
#     (
#         specs["partner_trans_mat"],
#         specs["n_partner_states"],
#     ) = read_in_partner_transition_specs(path_dict, specs)

#     specs["job_sep_probs"] = jnp.asarray(
#         pkl.load(open(path_dict["est_results"] + "job_sep_probs.pkl", "rb"))
#     )

#     # Set initial experience
#     specs["max_exp_diffs_per_period"] = create_max_experience(
#         path_dict, specs, load_precomputed
#     )
#     return specs


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
