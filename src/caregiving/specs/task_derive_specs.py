"""This module reads in specs and adds derived and estimated specs."""

import pickle as pkl
from pathlib import Path
from typing import Annotated, Any, Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.specs.experience_specs import create_max_experience
from caregiving.specs.family_specs import (
    predict_children_by_state,
    read_in_partner_transition_specs,
)
from caregiving.specs.income_specs import add_income_specs


def task_write_specs(
    path_to_specs: Path = SRC / "specs.yaml",
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
    path_to_child_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "nb_children_estimates.csv",
    path_to_partner_trans_mat: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_transition_matrix.csv",
    path_to_job_separation_probs: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_probs.pkl",
    path_to_struct_estimation_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
) -> Dict[str, Any]:
    """Read in specs and add specs from first-step estimation."""
    specs = read_and_derive_specs(path_to_specs)

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
    child_params = pd.read_csv(path_to_child_params, index_col=[0, 1, 2])

    # Matches Bruno's & Max's result
    specs["children_by_state"] = predict_children_by_state(child_params, specs)

    # Read in family transitions
    partner_trans_prop = pd.read_csv(
        path_to_partner_trans_mat, index_col=[0, 1, 2, 3, 4]
    )["proportion"]
    (
        specs["partner_trans_mat"],
        specs["n_partner_states"],
    ) = read_in_partner_transition_specs(partner_trans_prop, specs)

    specs["job_sep_probs"] = jnp.asarray(
        pkl.load(path_to_job_separation_probs.open("rb"))
    )

    # Set initial experience
    data_decision = pd.read_csv(path_to_struct_estimation_sample)
    specs["max_exp_diffs_per_period"] = create_max_experience(data_decision, specs)

    return specs
