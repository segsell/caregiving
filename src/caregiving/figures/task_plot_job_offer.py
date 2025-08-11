"""Job separations plots."""

import pickle as pkl
from pathlib import Path
from typing import Annotated, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.shared import SEX, UNEMPLOYED_CHOICES, WORK_CHOICES


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
    fig.tight_layout()
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


# def task_plot_job_offer_probs(
#     path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_struct_sample: Path = BLD
#     / "data"
#     / "soep_structural_estimation_sample.csv",
#     path_to_save: Annotated[Path, Product] = (
#         BLD / "plots" / "stochastic_processes" / "job_offer.png"
#     ),
#     male: bool = False,
# ):
#     """
#     Plot estimated and observed job-offer probabilities by age
#     (i.e., probability that an unemployed person starts work next period).

#     • Solid lines  = model-implied probabilities  (`specs["job_offer_probs"]`)
#     • Dashed lines = empirical shares inferred from the SOEP sample
#     • Colour       = education level (0 / 1)
#     """

#     # ---------- 1. Load model specs and observed data --------------------
#     with path_to_full_specs.open("rb") as file:
#         specs = pkl.load(file)

#     df = pd.read_csv(path_to_struct_sample)

#     # ---------- 2. Build empirical job-offer indicator -------------------
#     unemployed_vals = np.asarray(UNEMPLOYED_CHOICES).ravel().tolist()
#     work_vals = np.asarray(WORK_CHOICES).ravel().tolist()

#     df_unemp = df[df["lagged_choice"].isin(unemployed_vals)].copy()
#     df_unemp["work_start"] = df_unemp["choice"].isin(work_vals).astype(float)

#     obs_offer = df_unemp.groupby(["sex", "education", "age"])[
#         "work_start"
#     ].mean()  # MultiIndex Series

#     # ---------- 3. Axis set-up ------------------------------------------
#     n_working_periods = 65 - specs["start_age"] + 1
#     working_ages = np.arange(n_working_periods) + specs["start_age"]

#     sexes_to_plot = [1] if not male else [0, 1]
#     ncols = len(sexes_to_plot)

#     fig, axs = plt.subplots(
#         ncols=ncols,
#         figsize=(6 * ncols, 5),
#         squeeze=False,
#     )
#     axs = axs[0]

#     # ---------- 4. Loop over sexes / education ---------------------------
#     for col_idx, sex_var in enumerate(sexes_to_plot):
#         ax = axs[col_idx]
#         sex_label = specs["sex_labels"][sex_var]

#         for edu_var, edu_label in enumerate(specs["education_labels"]):

#             # ---- model-implied -----------------------------------------
#             est_series = specs["job_offer_probs"][  # <-- here
#                 sex_var, edu_var, :n_working_periods  # <-- here
#             ]

#             ax.plot(
#                 working_ages,
#                 est_series,
#                 label=f"Est. {edu_label}",
#                 color=JET_COLOR_MAP[edu_var],
#             )

#             # ---- empirical --------------------------------------------
#             obs_series = obs_offer.loc[(sex_var, edu_var, working_ages)]
#             ax.plot(
#                 working_ages,
#                 obs_series,
#                 label=f"Obs. {edu_label}",
#                 linestyle="--",
#                 color=JET_COLOR_MAP[edu_var],
#             )

#         if male:
#             ax.set_title(sex_label)
#         ax.set_xlabel("Age")
#         ax.set_ylabel("Share")
#         ax.set_ylim(0, 0.4)  # job-offer probs are larger than sep. probs
#         ax.set_xlim(specs["start_age"] - 0.5, 65 + 0.5)

#     axs[0].legend(loc="upper left")
#     fig.tight_layout()
#     fig.savefig(path_to_save, dpi=300)
#     plt.close(fig)


def _logistic(x: np.ndarray | float) -> np.ndarray | float:
    """Numeric helper."""
    return 1.0 / (1.0 + np.exp(-x))


def _read_offer_params(path: Path) -> Dict[str, float]:
    """job_offer_params.csv  →  dict {param_name: value}."""
    df = pd.read_csv(path)
    return dict(zip(df["param"], df["value"], strict=False))


def _model_offer_prob(age, edu, params, sex_suffix):
    """
    Logistic P(job offer | unemployed) for one (age, edu, sex).

    Parameters follow the naming pattern
       job_finding_logit_const_<suffix>
       job_finding_logit_age_<suffix>
       job_finding_logit_high_educ_<suffix>
    """
    const = params[f"job_finding_logit_const_{sex_suffix}"]
    beta_age = params[f"job_finding_logit_age_{sex_suffix}"]
    beta_edu = params[f"job_finding_logit_high_educ_{sex_suffix}"]
    # beta_age_squared = params[f"job_finding_logit_age_squared_{sex_suffix}"]
    # beta_age_cubed = params[f"job_finding_logit_age_cubed_{sex_suffix}"]
    x = (
        const
        + beta_age * age
        # + beta_age_squared * age**2
        # + beta_age_cubed * age**3
        + beta_edu * edu
    )
    # x = const + beta_age * age + beta_age_squared * age**2 + beta_edu * edu
    return _logistic(x)


def task_plot_job_offer_probs(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_struct_sample: Path = BLD
    / "data"
    / "soep_structural_estimation_sample.csv",
    path_to_job_offer_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_offer_params.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "plots" / "stochastic_processes" / "job_offer.png"
    ),
    male: bool = False,
):
    """
    Plot *job-offer/job-finding* probabilities by age, sex, education.

    Solid   = probabilities implied by the estimated logit parameters
    Dashed  = empirical shares (SOEP)
    Colour  = education level (0 = low, 1 = high)
    """

    # ------------ 1. inputs ------------------------------------------------
    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    params = _read_offer_params(path_to_job_offer_params)

    df = pd.read_csv(path_to_struct_sample)

    # ------------ 2. empirical shares -------------------------------------
    unemp_vals = np.asarray(UNEMPLOYED_CHOICES).ravel().tolist()
    work_vals = np.asarray(WORK_CHOICES).ravel().tolist()

    df_unemp = df[df["lagged_choice"].isin(unemp_vals)].copy()
    df_unemp["work_start"] = df_unemp["choice"].isin(work_vals).astype(float)

    obs_offer = df_unemp.groupby(["sex", "education", "age"])[
        "work_start"
    ].mean()  # MultiIndex (sex, edu, age)

    # ------------ 3. model probabilities ----------------------------------
    n_working_periods = 65 - specs["start_age"] + 1
    ages_grid = np.arange(n_working_periods) + specs["start_age"]

    sexes_to_plot = [1] if not male else [0, 1]
    ncols = len(sexes_to_plot)

    fig, axs = plt.subplots(
        ncols=ncols,
        figsize=(6 * ncols, 5),
        squeeze=False,
    )
    axs = axs[0]

    for col_idx, sex_var in enumerate(sexes_to_plot):
        ax = axs[col_idx]
        sex_suffix = specs["sex_labels"][sex_var].lower()  # 'men' or 'women'
        sex_label = specs["sex_labels"][sex_var]

        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # ---- model line ------------------------------------------------
            est_probs = np.array(
                [
                    _model_offer_prob(age, edu_var, params, sex_suffix)
                    for age in ages_grid
                ]
            )

            ax.plot(
                ages_grid,
                est_probs,
                color=JET_COLOR_MAP[edu_var],
                label=f"Est. {edu_label}",
            )

            # ---- empirical line -------------------------------------------
            obs_series = obs_offer.loc[(sex_var, edu_var, ages_grid)]
            ax.plot(
                ages_grid,
                obs_series,
                color=JET_COLOR_MAP[edu_var],
                linestyle="--",
                label=f"Obs. {edu_label}",
            )

        if male:
            ax.set_title(sex_label)
        ax.set_xlabel("Age")
        ax.set_ylabel("Share")
        ax.set_ylim(0, 0.7)
        ax.set_xlim(specs["start_age"] - 0.5, 65 + 0.5)

    axs[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)
