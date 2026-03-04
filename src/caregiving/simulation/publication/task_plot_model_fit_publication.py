"""Publication-style model fit plots: one PDF per labor outcome per sample (12 plots)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import PUBLICATION_PLOT_STYLE
from caregiving.model.shared import (
    DEAD,
    INFORMAL_CARE,
    RETIREMENT,
    UNEMPLOYED,
    PART_TIME,
    FULL_TIME,
    RETIREMENT_CHOICES,
    UNEMPLOYED_CHOICES,
    PART_TIME_CHOICES,
    FULL_TIME_CHOICES,
    SEX,
)
from caregiving.moments.task_create_soep_moments import (
    create_df_caregivers,
    create_df_non_caregivers,
    create_df_with_caregivers,
)

# Outcome index → y-axis label for publication
_Y_LABELS = {
    0: "Share retired",
    1: "Share non-employed",
    2: "Share working part-time",
    3: "Share full-time",
}

_OUTCOMES = ["retired", "non_employed", "part_time", "full_time"]
_SAMPLES = ["all", "non_caregiver", "caregiver"]


def _add_choice_group(data_sim: pd.DataFrame, data_emp: pd.DataFrame) -> None:
    """Map raw choice codes to 4-way choice_group (0=retirement, 1=unemployed, 2=part-time, 3=full-time). Modifies dataframes in place."""
    choice_groups_sim = {
        0: RETIREMENT,
        1: UNEMPLOYED,
        2: PART_TIME,
        3: FULL_TIME,
    }
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }
    data_sim["choice_group"] = np.nan
    data_emp["choice_group"] = np.nan
    for agg_code, raw_codes in choice_groups_sim.items():
        data_sim.loc[
            data_sim["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code
    for agg_code, raw_codes in choice_groups_emp.items():
        data_emp.loc[
            data_emp["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code
    data_sim["choice_group"] = data_sim["choice_group"].astype(int)
    data_emp["choice_group"] = data_emp["choice_group"].astype(int)


def _shares_by_age(
    data_emp: pd.DataFrame,
    data_sim: pd.DataFrame,
    choice_var: int,
    age_min: int,
    age_max: int,
):
    """Return ages, simulated share, empirical share for one outcome (choice_var 0–3)."""
    ages = list(range(age_min, age_max + 1))
    emp_sex = data_emp[data_emp["sex"] == SEX]
    sim_shares = (
        data_sim.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    emp_shares = (
        emp_sex.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    # Ensure all choice groups 0-3 exist (some samples may have no retirees etc.)
    sim_shares = sim_shares.reindex(columns=[0, 1, 2, 3], fill_value=0)
    emp_shares = emp_shares.reindex(columns=[0, 1, 2, 3], fill_value=0)
    vals_sim = sim_shares.reindex(ages, fill_value=0)[choice_var]
    vals_emp = emp_shares.reindex(ages, fill_value=0)[choice_var]
    return ages, vals_sim, vals_emp


@pytask.mark.publication_model_fit
def task_plot_model_fit_publication(  # noqa: PLR0915
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
    path_to_share_retired_all: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_retired_all.pdf",
    path_to_share_retired_non_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_retired_non_caregiver.pdf",
    path_to_share_retired_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_retired_caregiver.pdf",
    path_to_share_non_employed_all: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_non_employed_all.pdf",
    path_to_share_non_employed_non_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_non_employed_non_caregiver.pdf",
    path_to_share_non_employed_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_non_employed_caregiver.pdf",
    path_to_share_part_time_all: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_part_time_all.pdf",
    path_to_share_part_time_non_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_part_time_non_caregiver.pdf",
    path_to_share_part_time_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_part_time_caregiver.pdf",
    path_to_share_full_time_all: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_full_time_all.pdf",
    path_to_share_full_time_non_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_full_time_non_caregiver.pdf",
    path_to_share_full_time_caregiver: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "model_fit"
    / "share_full_time_caregiver.pdf",
) -> None:
    """Produce 12 publication PDFs: one per (outcome, sample) with thesis-style formatting."""
    with open(path_to_specs, "rb") as f:
        specs = pickle.load(f)

    start_year = 2001
    end_year = 2019
    end_age_msm = specs["end_age_msm"]
    start_age = specs["start_age"]
    start_age_caregivers = specs["start_age_caregiving"]
    end_age_caregiver = 69

    df_emp_full = pd.read_csv(path_to_empirical_data, index_col=[0])
    df_caregivers_full = pd.read_csv(path_to_caregivers_sample, index_col=[0])

    df_emp_all = create_df_with_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age_msm,
    )
    df_emp_non_caregiver = create_df_non_caregivers(
        df_full=df_emp_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age_msm,
    )
    df_emp_caregiver = create_df_caregivers(
        df_caregivers_full=df_caregivers_full,
        specs=specs,
        start_year=start_year,
        end_year=end_year,
        end_age=end_age_msm,
    )

    df_sim = pd.read_pickle(path_to_simulated_data).reset_index()
    df_sim["sex"] = SEX
    df_sim["age"] = df_sim["period"] + start_age
    df_sim = df_sim[df_sim["health"] != DEAD].copy()

    df_sim_all = df_sim.copy()
    df_sim_non_caregiver = df_sim.loc[
        ~df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()
    df_sim_caregiver = df_sim.loc[
        df_sim["choice"].isin(np.asarray(INFORMAL_CARE).tolist())
    ].copy()

    _add_choice_group(df_sim_all, df_emp_all)
    _add_choice_group(df_sim_non_caregiver, df_emp_non_caregiver)
    _add_choice_group(df_sim_caregiver, df_emp_caregiver)

    emp_by_sample = {
        "all": df_emp_all,
        "non_caregiver": df_emp_non_caregiver,
        "caregiver": df_emp_caregiver,
    }
    sim_by_sample = {
        "all": df_sim_all,
        "non_caregiver": df_sim_non_caregiver,
        "caregiver": df_sim_caregiver,
    }
    age_range_by_sample = {
        "all": (start_age, end_age_msm),
        "non_caregiver": (start_age, end_age_msm),
        "caregiver": (start_age_caregivers, end_age_caregiver),
    }

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    products = [
        (0, "retired", "all", path_to_share_retired_all),
        (0, "retired", "non_caregiver", path_to_share_retired_non_caregiver),
        (0, "retired", "caregiver", path_to_share_retired_caregiver),
        (1, "non_employed", "all", path_to_share_non_employed_all),
        (1, "non_employed", "non_caregiver", path_to_share_non_employed_non_caregiver),
        (1, "non_employed", "caregiver", path_to_share_non_employed_caregiver),
        (2, "part_time", "all", path_to_share_part_time_all),
        (2, "part_time", "non_caregiver", path_to_share_part_time_non_caregiver),
        (2, "part_time", "caregiver", path_to_share_part_time_caregiver),
        (3, "full_time", "all", path_to_share_full_time_all),
        (3, "full_time", "non_caregiver", path_to_share_full_time_non_caregiver),
        (3, "full_time", "caregiver", path_to_share_full_time_caregiver),
    ]

    for choice_var, _outcome_name, sample_name, path_to_save in products:
        age_min, age_max = age_range_by_sample[sample_name]
        data_emp = emp_by_sample[sample_name]
        data_sim = sim_by_sample[sample_name]
        ages, vals_sim, vals_emp = _shares_by_age(
            data_emp, data_sim, choice_var, age_min, age_max
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(
            ages,
            vals_sim,
            color="0",
            linestyle="-",
            linewidth=style["linewidth"],
        )
        ax.plot(
            ages,
            vals_emp,
            color="0.5",
            linestyle="--",
            linewidth=style["linewidth"],
        )
        ax.set_xlim(age_min, age_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Age", fontsize=style["label_fontsize"])
        ax.set_ylabel(_Y_LABELS[choice_var], fontsize=style["label_fontsize"])
        ax.tick_params(
            axis="both",
            labelsize=style["xtick_fontsize"],
            length=style["tick_length"],
            width=style["tick_width"],
        )
        ax.grid(
            True, axis="y", alpha=style["grid_alpha"], linewidth=style["grid_linewidth"]
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        if path_to_save.suffix.lower() == ".pdf":
            _pdf_fonttype = plt.rcParams["pdf.fonttype"]
            plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(
            path_to_save,
            dpi=style["savefig_dpi"],
            bbox_inches="tight",
            pad_inches=style["savefig_pad_inches"],
        )
        if path_to_save.suffix.lower() == ".pdf":
            plt.rcParams["pdf.fonttype"] = _pdf_fonttype
        plt.close(fig)
