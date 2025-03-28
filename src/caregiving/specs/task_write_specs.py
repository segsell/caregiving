"""This module reads in specs and adds derived and estimated specs."""

import pickle as pkl
from pathlib import Path
from typing import Annotated, Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.specs.experience_specs import create_max_experience
from caregiving.specs.family_specs import (
    predict_age_of_youngest_child_by_state,
    predict_children_by_state,
    read_in_partner_transition_specs,
)
from caregiving.specs.health_specs import read_in_health_transition_specs
from caregiving.specs.income_specs import add_income_specs

jax.config.update("jax_enable_x64", True)


def task_write_specs(
    path_to_load_specs: Path = SRC / "specs.yaml",
    path_to_wage_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "wage_eq_params.csv",
    path_to_partner_wage_params_men: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_wage_eq_params_men.csv",
    path_to_partner_wage_params_women: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_wage_eq_params_women.csv",
    path_to_pop_avg_working_hours: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "pop_avg_working_hours.csv",
    path_to_pop_avg_annual_wage: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "pop_avg_annual_wage.npy",
    path_to_nb_child_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "nb_children_estimates.csv",
    path_to_age_youngest_child_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "age_youngest_child.csv",
    path_to_partner_trans_mat: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_transition_matrix.csv",
    path_to_health_transition_mat: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix.csv",
    path_to_mortality_transition_mat: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix.csv",
    path_to_job_separation_probs: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_probs.pkl",
    path_to_struct_estimation_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_save_max_exp_diff: Annotated[Path, Product] = BLD
    / "model"
    / "specs"
    / "max_exp_diffs_per_period.txt",
    path_to_save_specs_dict: Annotated[Path, Product] = BLD
    / "model"
    / "specs"
    / "specs_full.pkl",
) -> Dict[str, Any]:
    """Read in specs and add specs from first-step estimation."""
    specs = read_and_derive_specs(path_to_load_specs)

    # Add income specs
    wage_params = pd.read_csv(path_to_wage_params, index_col=0)
    partner_wage_params_men = pd.read_csv(path_to_partner_wage_params_men)
    partner_wage_params_women = pd.read_csv(path_to_partner_wage_params_women)
    avg_working_hours = pd.read_csv(path_to_pop_avg_working_hours)  # slightly different
    mean_annual_wage = np.load(path_to_pop_avg_annual_wage)  # identical :)

    specs = add_income_specs(
        specs,
        wage_params=wage_params,
        partner_wage_params_men=partner_wage_params_men,
        partner_wage_params_women=partner_wage_params_women,
        avg_working_hours=avg_working_hours,
        mean_annual_wage=mean_annual_wage,
    )

    # family transitions
    children_params = pd.read_csv(path_to_nb_child_params, index_col=[0, 1, 2])

    age_youngest_child_params = pd.read_csv(
        path_to_age_youngest_child_params, index_col=[0, 1, 2]
    )

    # Matches Bruno's & Max's result
    specs["children_by_state"] = predict_children_by_state(children_params, specs)
    specs["child_age_youngest_by_state"] = predict_age_of_youngest_child_by_state(
        age_youngest_child_params,
        specs,
    )

    # Read in family transitions
    partner_trans_prop = pd.read_csv(
        path_to_partner_trans_mat, index_col=[0, 1, 2, 3, 4]
    )["proportion"]
    (
        specs["partner_trans_mat"],
        specs["n_partner_states"],
    ) = read_in_partner_transition_specs(partner_trans_prop, specs)

    # Read in health transition matrix
    health_trans_probs_df = pd.read_csv(
        path_to_health_transition_mat,
    )
    death_prob_df = pd.read_csv(path_to_mortality_transition_mat)
    specs["health_trans_mat"] = read_in_health_transition_specs(
        health_trans_probs_df, death_prob_df, specs
    )

    # if "health_vars_three" in specs.keys():
    #     health_trans_probs_df = pd.read_csv(
    #         path_to_health_transition_mat_three_states,
    #     )
    #     death_prob_df = pd.read_csv(path_to_mortality_transition_mat_three_states)
    #     specs["health_trans_mat_three"] = read_in_health_transition_specs(
    #         health_trans_probs_df, death_prob_df, specs
    #     )

    specs["job_sep_probs"] = jnp.asarray(
        pkl.load(path_to_job_separation_probs.open("rb"))
    )

    # Set initial experience
    data_decision = pd.read_csv(path_to_struct_estimation_sample)

    specs["max_exp_diffs_per_period"] = create_max_experience(
        data_decision, specs, path_to_save_txt=path_to_save_max_exp_diff
    )

    with path_to_save_specs_dict.open("wb") as f:
        pkl.dump(specs, f)

    return specs
