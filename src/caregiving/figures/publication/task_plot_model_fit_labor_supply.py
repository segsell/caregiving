"""Plot labor supply model fit between empirical and simulated data."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.figures.publication.plotting_functions import (
    plot_choice_shares_by_education_bw,
)
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    FULL_TIME_CHOICES,
    INFORMAL_CARE,
    PART_TIME,
    PART_TIME_CHOICES,
    RETIREMENT,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CHOICES,
    WORK_CHOICES,
)
from caregiving.moments.task_create_soep_moments import (
    create_df_caregivers,
    create_df_non_caregivers,
    create_df_with_caregivers,
)


@pytask.mark.publication
def task_plot_model_fit_labor_supply(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_empirical_data: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_all_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "labor_supply_fit_all.png",
    path_to_save_all_combined_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "labor_supply_fit_all_combined.png",
    # path_to_save_non_caregivers_plot: Annotated[Path, Product] = BLD
    # / "figures"
    # / "publication"
    # / "labor_supply_fit_non_caregivers.png",
    # path_to_save_caregivers_plot: Annotated[Path, Product] = BLD
    # / "figures"
    # / "publication"
    # / "labor_supply_fit_caregivers.png",
) -> None:
    """Plot labor supply model fit for all, non-caregivers, and caregivers.

    Creates three plots showing choice shares by age and education:
    1. All individuals
    2. Non-caregivers
    3. Caregivers

    All plots are in black and white with observed lines dashed and
    simulated lines black solid.

    """

    specs = pickle.load(path_to_specs.open("rb"))

    # Load full datasets
    df_emp_full = pd.read_csv(path_to_empirical_data, index_col=[0])
    # df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])

    # Load simulated data
    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX
    df_sim["age"] = df_sim["period"] + specs["start_age"]
    df_sim = df_sim[df_sim["health"] != DEAD].copy()

    # Create standardized subsamples
    start_year = 2001
    end_year = 2019
    end_age = specs["end_age_msm"]

    # All individuals (with caregivers)
    df_emp_all = create_df_with_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age,
    )

    # Non-caregivers (commented out - not currently used)
    # df_emp_non_caregivers = create_df_non_caregivers(
    #     df_full=df_emp_full,
    #     specs=specs,
    #     start_year=start_year,
    #     end_year=end_year,
    #     end_age=end_age,
    # )

    # Caregivers (commented out - not currently used)
    # df_emp_caregivers = create_df_caregivers(
    #     df_caregivers_full=df_caregivers_full,
    #     specs=specs,
    #     start_year=start_year,
    #     end_year=end_year,
    #     end_age=end_age,
    # )

    # Simulated caregivers (commented out - not currently used)
    # df_sim_caregivers = df_sim.loc[
    #     df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    # ]

    # Plot 1: All individuals (separate subplots for each education)
    plot_choice_shares_by_education_bw(
        data_emp=df_emp_all,
        data_sim=df_sim,
        specs=specs,
        path_to_save_plot=path_to_save_all_plot,
    )

    # # Plot 1b: All individuals (combined education levels in one plot)
    # plot_choice_shares_combined_education_bw(
    #     data_emp=df_emp_all,
    #     data_sim=df_sim,
    #     specs=specs,
    #     path_to_save_plot=path_to_save_all_combined_plot,
    # )

    # # Plot 2: Non-caregivers
    # plot_choice_shares_by_education_bw(
    #     data_emp=df_emp_non_caregivers,
    #     data_sim=df_sim,
    #     specs=specs,
    #     path_to_save_plot=path_to_save_non_caregivers_plot,
    # )

    # # Plot 3: Caregivers
    # plot_choice_shares_by_education_bw(
    #     data_emp=df_emp_caregivers,
    #     data_sim=df_sim_caregivers,
    #     specs=specs,
    #     path_to_save_plot=path_to_save_caregivers_plot,
    # )
