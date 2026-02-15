"""Create restricted SOEP moments: wealth-only and full with mean wealth by age bin."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

import dcegm
from caregiving.config import BLD
from caregiving.model.shared import SEX
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.moments.task_create_soep_moments import (
    compute_mean_by_age_bin,
    compute_median_by_age_bin,
    create_df_wealth,
)


@pytask.mark.moments
@pytask.mark.soep_moments
def task_create_soep_moments_wealth_only(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "moments_wealth_only.csv",
    path_to_save_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "variances_wealth_only.csv",
) -> None:
    """Create only median wealth by age bin moments (and their variances)."""
    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model_class = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )

    start_age = specs["start_age"]
    age_range_wealth = range(start_age, specs["end_age_wealth"] + 10)

    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    df_wealth = create_df_wealth(df_full, model_class=model_class)
    df_wealth = df_wealth[df_wealth["sex"] == SEX].copy()

    df_wealth_low = df_wealth[df_wealth["education"] == 0].copy()
    df_wealth_high = df_wealth[df_wealth["education"] == 1].copy()

    moments = {}
    variances = {}

    moments, variances = compute_median_by_age_bin(
        df_wealth_low,
        moments,
        variances,
        wealth_var="adjusted_wealth",
        age_range=age_range_wealth,
        label="wealth_low_education",
    )
    moments, variances = compute_median_by_age_bin(
        df_wealth_high,
        moments,
        variances,
        wealth_var="adjusted_wealth",
        age_range=age_range_wealth,
        label="wealth_high_education",
    )

    moments_df = pd.DataFrame({"value": pd.Series(moments)})
    moments_df.index.name = "moment"
    variances_df = pd.DataFrame({"value": pd.Series(variances)})
    variances_df.index.name = "moment"

    path_to_save_moments.parent.mkdir(parents=True, exist_ok=True)
    moments_df.to_csv(path_to_save_moments, index=True)
    variances_df.to_csv(path_to_save_variances, index=True)


@pytask.mark.moments
@pytask.mark.soep_moments
def task_create_soep_moments_wealth_mean(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_model_config: Path = BLD / "model" / "model_config.pkl",
    path_to_model: Path = BLD / "model" / "model.pkl",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_full_moments: Path = BLD / "moments" / "moments_full.csv",
    path_to_full_variances: Path = BLD / "moments" / "variances_full.csv",
    path_to_save_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "moments_full_with_mean_wealth.csv",
    path_to_save_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "variances_full_with_mean_wealth.csv",
) -> None:
    """Create full set of moments but with mean wealth by age bin instead of median.

    Loads the full moments/variances from the main task (soep_moments_new.csv and
    soep_variances_new.csv), drops median wealth-by-age-bin keys, adds mean
    wealth-by-age-bin moments (and variances) from data, and saves.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    model_config = pickle.load(path_to_model_config.open("rb"))

    model_class = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=path_to_model,
    )

    start_age = specs["start_age"]
    age_range_wealth = range(start_age, specs["end_age_wealth"] + 10)
    age_min = int(np.min(age_range_wealth))
    age_max = int(np.max(age_range_wealth))
    bin_edges_w = list(range(age_min, age_max + 6, 5))
    bin_labels_w = [f"{s}_{s+4}" for s in bin_edges_w[:-1]]
    age_bins_wealth = (bin_edges_w, bin_labels_w)

    # Load full moments and variances from main task
    moments_df = pd.read_csv(path_to_full_moments, index_col=[0])
    variances_df = pd.read_csv(path_to_full_variances, index_col=[0])
    moments = moments_df["value"].to_dict()
    variances = variances_df["value"].to_dict()

    # Remove median wealth by age bin keys (to replace with mean)
    median_wealth_prefixes = (
        "median_wealth_low_education_wealth_age_bin_",
        "median_wealth_high_education_wealth_age_bin_",
    )
    var_wealth_prefixes = (
        "var_wealth_low_education_wealth_age_bin_",
        "var_wealth_high_education_wealth_age_bin_",
    )
    moments = {
        k: v for k, v in moments.items() if not k.startswith(median_wealth_prefixes)
    }
    variances = {
        k: v for k, v in variances.items() if not k.startswith(var_wealth_prefixes)
    }

    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    df_wealth = create_df_wealth(
        df_full, model_class=model_class, trim_upper_wealth_quantile=True
    )
    df_wealth = df_wealth[df_wealth["sex"] == SEX].copy()

    df_wealth_low = df_wealth[df_wealth["education"] == 0].copy()
    df_wealth_high = df_wealth[df_wealth["education"] == 1].copy()

    moments, variances = compute_mean_by_age_bin(
        df_wealth_low,
        moments,
        variances,
        variable="adjusted_wealth",
        age_bins=age_bins_wealth,
        label="wealth_low_education",
    )
    moments, variances = compute_mean_by_age_bin(
        df_wealth_high,
        moments,
        variances,
        variable="adjusted_wealth",
        age_bins=age_bins_wealth,
        label="wealth_high_education",
    )

    # Put mean_wealth moments (and their variances) first in the output
    mean_wealth_prefixes = (
        "mean_wealth_low_education_",
        "mean_wealth_high_education_",
    )
    var_wealth_prefixes = (
        "var_wealth_low_education_",
        "var_wealth_high_education_",
    )
    mean_keys = [k for k in moments if k.startswith(mean_wealth_prefixes)]
    other_moment_keys = [k for k in moments if not k.startswith(mean_wealth_prefixes)]
    var_wealth_keys = [k for k in variances if k.startswith(var_wealth_prefixes)]
    other_var_keys = [k for k in variances if not k.startswith(var_wealth_prefixes)]
    moments_ordered = {k: moments[k] for k in mean_keys + other_moment_keys}
    variances_ordered = {k: variances[k] for k in var_wealth_keys + other_var_keys}

    moments_df = pd.DataFrame({"value": pd.Series(moments_ordered)})
    moments_df.index.name = "moment"
    variances_df = pd.DataFrame({"value": pd.Series(variances_ordered)})
    variances_df.index.name = "moment"

    path_to_save_moments.parent.mkdir(parents=True, exist_ok=True)
    moments_df.to_csv(path_to_save_moments, index=True)
    variances_df.to_csv(path_to_save_variances, index=True)
