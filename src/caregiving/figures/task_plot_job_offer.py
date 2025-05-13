"""Job separations plots."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC


def task_plot_job_transitions(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_job_separation_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "job_separation.png",
    male: bool = False,
):
    """Plot job separation probabilities."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    n_working_periods = 65 - specs["start_age"] + 1

    # df_job = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    df_job = pd.read_csv(path_to_data)

    obs_shares = df_job.groupby(["sex", "education", "age"])["job_sep"].mean()
    working_ages = np.arange(n_working_periods) + specs["start_age"]

    # n_education_types = specs["n_education_types"]
    # n_sexes = specs["n_sexes"]
    # job_offer_probs = np.zeros(
    #     (n_sexes, n_education_types, n_working_periods), dtype=float
    # )

    # for sex in range(n_sexes):
    #     for edu in range(n_education_types):
    #         for period in range(n_working_periods):
    # job_offer_probs[sex, edu, period] = job_offer_process_transition(
    #     params=params,
    #     options=specs,
    #     sex=sex,
    #     education=edu,
    #     period=period,
    #     choice=1,
    # )[1]

    # -----------------------------------------------------------------
    sexes_to_plot = [1] if not male else [0, 1]
    ncols = len(sexes_to_plot)

    fig, axs = plt.subplots(
        ncols=ncols,
        figsize=(6 * ncols, 5),
        squeeze=False,
    )
    axs = axs[0]  # flatten

    # fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #     ax = axs[sex_var]
    #     for edu_var, edu_label in enumerate(specs["education_labels"]):
    for col_idx, sex_var in enumerate(sexes_to_plot):
        ax = axs[col_idx]
        sex_label = specs["sex_labels"][sex_var]

        for edu_var, edu_label in enumerate(specs["education_labels"]):

            ax.plot(
                working_ages,
                specs["job_sep_probs"][sex_var, edu_var, :n_working_periods],
                label=f"Est. {edu_label}",
                color=JET_COLOR_MAP[edu_var],
            )
            ax.plot(
                working_ages,
                obs_shares.loc[(sex_var, edu_var, working_ages)],
                label=f"Obs. {edu_label}",
                linestyle="--",
                color=JET_COLOR_MAP[edu_var],
            )

        if male:
            ax.set_title(str(sex_label))
        ax.set_xlabel("Age")
        ax.set_ylabel("Share")
        ax.set_ylim([0, 0.1])

    axs[0].legend(loc="upper left")
    fig.savefig(path_to_save)
