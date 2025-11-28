"""This module reads in specs and adds derived and estimated specs."""

import pickle as pkl
from pathlib import Path
from typing import Annotated, Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.caregiving_specs import (
    convert_adl_state_transition_array_to_df,
    read_in_adl_state_transition_specs_with_death,
    read_in_care_supply_transition_specs,
    read_in_mother_age_diff_specs,
)
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.specs.experience_specs import create_max_experience
from caregiving.specs.family_specs import (
    predict_age_of_youngest_child_by_state,
    predict_children_by_state,
    read_in_partner_transition_specs,
)
from caregiving.specs.health_specs import (
    plot_health_death_transitions_good_bad,
    read_in_health_transition_specs,
    read_in_health_transition_specs_df,
)
from caregiving.specs.income_specs import add_income_specs
from caregiving.utils import create_age_bins

jax.config.update("jax_enable_x64", True)


@pytask.mark.specs
def task_write_specs(
    path_to_load_specs: Path = SRC / "specs.yaml",
    path_to_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
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
    / "mortality_transition_matrix_logit.csv",
    path_to_health_transition_mat_good_medium_bad: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_good_medium_bad.csv",
    path_to_adl_state_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_state_transition_matrix.csv",
    path_to_survival_by_age: Path = BLD
    / "model"
    / "initial_conditions"
    / "survival_by_age.csv",
    path_to_exogenous_care_supply_transition: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "exogenous_care_supply_transition_matrix.csv",
    path_to_job_separation_probs: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_sep_probs.pkl",
    path_to_struct_estimation_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_save_health_death_transition_matrix_good_bad: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_death_transition_matrix_good_bad.csv",
    path_to_save_health_death_transition_good_bad: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "health_death_transition_good_bad.png",
    path_to_save_max_exp_diff: Annotated[Path, Product] = BLD
    / "model"
    / "specs"
    / "max_exp_diffs_per_period.txt",
    path_to_save_specs_dict: Annotated[Path, Product] = BLD
    / "model"
    / "specs"
    / "specs_full.pkl",
    path_to_save_adl_state_transition_with_death: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_state_transition_matrix_with_death.csv",
) -> Dict[str, Any]:
    """Read in specs and add specs from first-step estimation."""

    estimation_sample = pd.read_csv(path_to_sample, index_col=[0])
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

    # Read in health transition matrix including death probabilities
    health_trans_probs_df = pd.read_csv(
        path_to_health_transition_mat,
    )
    death_prob_df = pd.read_csv(path_to_mortality_transition_mat)
    specs["health_trans_mat"] = read_in_health_transition_specs(
        health_trans_probs_df, death_prob_df, specs
    )

    health_death_trans_mat = read_in_health_transition_specs_df(
        health_trans_probs_df=health_trans_probs_df,
        death_prob_df=death_prob_df,
        specs=specs,
    )
    health_death_trans_mat.to_csv(path_to_save_health_death_transition_matrix_good_bad)
    plot_health_death_transitions_good_bad(
        specs=specs,
        df=health_death_trans_mat,
        path_to_save_plot=path_to_save_health_death_transition_good_bad,
    )

    # if "health_labels_three" in specs.keys():
    #     # Read in health transition matrix including death probabilities
    #     health_trans_probs_df = pd.read_csv(
    #         path_to_health_transition_mat_good_medium_bad,
    #     )
    #     death_prob_df = pd.read_csv(path_to_mortality_transition_mat_good_medium_bad)
    #     specs["health_trans_mat_three"] = (
    #         read_in_health_transition_specs_good_medium_bad(
    #             health_trans_probs_df, death_prob_df, specs
    #         )
    #     )

    #     health_death_trans_mat_three = (
    #         read_in_health_transition_specs_good_medium_bad_df(
    #             health_trans_probs_df=health_trans_probs_df,
    #             death_prob_df=death_prob_df,
    #             specs=specs,
    #         )
    #     )
    #     health_death_trans_mat_three.to_csv(
    #         path_to_save_health_death_transition_matrix_good_medium_bad
    #     )
    #     plot_health_death_transitions_good_medium_bad(
    #         specs=specs,
    #         df=health_death_trans_mat_three,
    #         path_to_save_plot=path_to_save_health_death_transition_good_medium_bad,
    #     )

    # =================================================================================
    if "adl_labels" in specs.keys():
        # # adl_labels = ["No limitations", "Limitations"]
        # adl_transitions = pd.read_csv(path_to_limitations_with_adl_transition)
        # specs["limitations_with_adl_mat"] = read_in_adl_transition_specs_binary(
        #     adl_transitions, specs
        # )

        # Read in ADL state transitions with death (no health dependence)
        adl_state_transitions = pd.read_csv(path_to_adl_state_transition_matrix)
        survival_by_age = pd.read_csv(path_to_survival_by_age, index_col=0).squeeze(
            "columns"
        )
        survival_by_age.index = survival_by_age.index.astype(int)
        adl_state_trans_mat_with_death = read_in_adl_state_transition_specs_with_death(
            adl_state_transitions, survival_by_age, specs
        )
        specs["adl_state_transition_mat_with_death"] = adl_state_trans_mat_with_death

        # Convert to DataFrame and save as CSV
        adl_state_trans_df_with_death = convert_adl_state_transition_array_to_df(
            adl_state_trans_mat_with_death, specs
        )
        adl_state_trans_df_with_death.to_csv(
            path_to_save_adl_state_transition_with_death, index=False
        )

        exogenous_care_supply = pd.read_csv(
            path_to_exogenous_care_supply_transition,
        )
        specs["exog_care_supply"] = read_in_care_supply_transition_specs(
            exogenous_care_supply, specs
        )
        # make sure output is uin8
        specs["mother_age_diff"] = read_in_mother_age_diff_specs(estimation_sample)

    # =================================================================================

    specs["job_sep_probs"] = jnp.asarray(
        pkl.load(path_to_job_separation_probs.open("rb"))
    )

    # Set initial experience
    data_decision = pd.read_csv(path_to_struct_estimation_sample)

    specs["max_exp_diffs_per_period"] = create_max_experience(
        data_decision, specs, path_to_save_txt=path_to_save_max_exp_diff
    )

    # Precompute age bins for moment calculation
    start_age_caregiving = specs["start_age_caregiving"]
    end_age_caregiving = specs["end_age_caregiving"]

    specs["age_bins_5year"] = create_age_bins(
        start_age_caregiving, end_age_caregiving, bin_size=5, min_remainder_size=5
    )
    specs["age_bins_3year"] = create_age_bins(
        start_age_caregiving, end_age_caregiving, bin_size=3, min_remainder_size=2
    )

    with path_to_save_specs_dict.open("wb") as f:
        pkl.dump(specs, f)

    return specs
