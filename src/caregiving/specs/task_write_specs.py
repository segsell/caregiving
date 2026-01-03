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
    read_in_adl_state_transition_specs,
    read_in_adl_state_transition_specs_light_intensive,
    read_in_adl_transition_specs,
    read_in_adl_transition_specs_binary,
    read_in_care_supply_transition_specs,
    read_in_death_transition_specs,
    read_in_mother_age_diff_specs,
    read_in_survival_by_age_specs,
    weight_adl_transitions_by_survival,
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
    plot_health_death_transitions_good_medium_bad,
    read_in_health_transition_specs,
    read_in_health_transition_specs_df,
    read_in_health_transition_specs_good_medium_bad,
    read_in_health_transition_specs_good_medium_bad_df,
)
from caregiving.specs.income_specs import add_income_specs
from caregiving.specs.inheritance_specs import (
    read_in_inheritance_amount_specs,
    read_in_inheritance_prob_specs,
)

jax.config.update("jax_enable_x64", True)


@pytask.mark.specs
def task_write_specs(  # noqa: PLR0915
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
    path_to_mortality_transition_mat_good_medium_bad: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_logit_good_medium_bad.csv",
    path_to_lifetable: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable.csv",
    path_to_limitations_with_adl_transition: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_transition_matrix.csv",
    path_to_adl_state_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_state_transition_matrix.csv",
    path_to_inheritance_prob_spec7_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_specs"
    / "spec7_any_care_this_year_filter_parent_this_year_params.csv",
    path_to_inheritance_amount_spec12_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_specs_two_care"
    / "spec12_care_recent_filter_parent_recent_params.csv",
    path_to_save_survival_by_age: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
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
    path_to_save_health_death_transition_matrix_good_medium_bad: Annotated[
        Path, Product
    ] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_death_transition_matrix_good_medium_bad.csv",
    path_to_save_health_death_transition_good_bad: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "health_death_transition_good_bad.png",
    path_to_save_health_death_transition_good_medium_bad: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "health_death_transition_good_medium_bad.png",
    path_to_save_max_exp_diff: Annotated[Path, Product] = BLD
    / "model"
    / "specs"
    / "max_exp_diffs_per_period.txt",
    path_to_save_specs_dict: Annotated[Path, Product] = BLD
    / "model"
    / "specs"
    / "specs_full.pkl",
    path_to_save_adl_state_transition_mat: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_state_transition_mat.csv",
    path_to_save_adl_state_transition_mat_light_intensive: Annotated[
        Path, Product
    ] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_state_transition_mat_light_intensive.csv",
    path_to_save_death_transition_mat: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "death_transition_mat.csv",
    path_to_save_inheritance_prob_mat: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_prob_matrix.csv",
    path_to_save_inheritance_amount_mat: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_matrix.csv",
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

    if "health_labels_three" in specs.keys():
        # Read in health transition matrix including death probabilities
        health_trans_probs_df = pd.read_csv(
            path_to_health_transition_mat_good_medium_bad,
        )
        death_prob_df = pd.read_csv(path_to_mortality_transition_mat_good_medium_bad)
        specs["health_trans_mat_three"] = (
            read_in_health_transition_specs_good_medium_bad(
                health_trans_probs_df, death_prob_df, specs
            )
        )

        health_death_trans_mat_three = (
            read_in_health_transition_specs_good_medium_bad_df(
                health_trans_probs_df=health_trans_probs_df,
                death_prob_df=death_prob_df,
                specs=specs,
            )
        )
        health_death_trans_mat_three.to_csv(
            path_to_save_health_death_transition_matrix_good_medium_bad
        )
        plot_health_death_transitions_good_medium_bad(
            specs=specs,
            df=health_death_trans_mat_three,
            path_to_save_plot=path_to_save_health_death_transition_good_medium_bad,
        )

    # Create and save survival by age for both men and women
    lifetable = pd.read_csv(path_to_lifetable)
    lifetable = lifetable.sort_values(["sex", "age"])  # ensure order
    lifetable["cum_survival_prob"] = (
        (1 - lifetable["death_prob"]).groupby(lifetable["sex"]).cumprod()
    )
    survival_by_age_df = lifetable[["sex", "age", "cum_survival_prob"]].copy()
    survival_by_age_df = survival_by_age_df.sort_values(["sex", "age"])
    survival_by_age_df.to_csv(path_to_save_survival_by_age, index=False)

    # Create death transition matrix from lifetable
    death_transition_df = lifetable[["sex", "age", "death_prob"]].copy()
    death_transition_df = death_transition_df.sort_values(["sex", "age"])
    specs["death_transition_mat"] = read_in_death_transition_specs(
        death_transition_df, specs, path_to_save=path_to_save_death_transition_mat
    )

    # Read survival by age into specs
    specs["survival_by_age_mat"] = read_in_survival_by_age_specs(
        survival_by_age_df, specs
    )
    # # Store minimum age for age-to-index conversion (same for death and survival)
    # specs["survival_min_age"] = min(survival_by_age_df["age"].unique())

    if "adl_labels" in specs.keys():
        # adl_labels = ["No limitations", "Limitations"]
        adl_transitions = pd.read_csv(path_to_limitations_with_adl_transition)
        specs["limitations_with_adl_mat"] = read_in_adl_transition_specs_binary(
            adl_transitions, specs
        )

        # Read ADL state transition matrix (simpler, no health dimension)
        adl_state_transitions = pd.read_csv(path_to_adl_state_transition_matrix)
        specs["adl_state_transition_mat"] = read_in_adl_state_transition_specs(
            adl_state_transitions,
            specs,
            path_to_save=path_to_save_adl_state_transition_mat,
        )

        # Read ADL state transition matrix with collapsed categories
        # (ADL 2 and ADL 3 collapsed into single "intensive" category)
        specs["adl_state_transition_mat_light_intensive"] = (
            read_in_adl_state_transition_specs_light_intensive(
                adl_state_transitions,
                specs,
                path_to_save=path_to_save_adl_state_transition_mat_light_intensive,
            )
        )

        # Weight ADL transitions by survival probabilities
        specs["adl_state_transition_mat_weighted"] = weight_adl_transitions_by_survival(
            specs
        )

        exogenous_care_supply = pd.read_csv(
            path_to_exogenous_care_supply_transition,
        )
        specs["exog_care_supply"] = read_in_care_supply_transition_specs(
            exogenous_care_supply, specs
        )
        # make sure output is uin8
        specs["mother_age_diff"] = read_in_mother_age_diff_specs(estimation_sample)

    specs["job_sep_probs"] = jnp.asarray(
        pkl.load(path_to_job_separation_probs.open("rb"))
    )

    # Set initial experience
    data_decision = pd.read_csv(path_to_struct_estimation_sample)

    specs["max_exp_diffs_per_period"] = create_max_experience(
        data_decision, specs, path_to_save_txt=path_to_save_max_exp_diff
    )

    # Load inheritance parameters
    inheritance_prob_spec7_params = pd.read_csv(
        path_to_inheritance_prob_spec7_params, index_col=0
    )
    inheritance_amount_spec12_params = pd.read_csv(
        path_to_inheritance_amount_spec12_params, index_col=0
    )
    specs["inheritance_prob_params"] = inheritance_prob_spec7_params
    specs["inheritance_amount_params"] = inheritance_amount_spec12_params

    # Precompute inheritance probability matrix by age, education, and care type
    specs["inheritance_prob_mat"] = read_in_inheritance_prob_specs(
        specs, path_to_save=path_to_save_inheritance_prob_mat
    )
    # Precompute inheritance amount matrix by age, education, and care type
    specs["inheritance_amount_mat"] = read_in_inheritance_amount_specs(
        specs, path_to_save=path_to_save_inheritance_amount_mat
    )

    with path_to_save_specs_dict.open("wb") as f:
        pkl.dump(specs, f)
    return specs
