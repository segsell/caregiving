"""SOEP-IS: Exogenous supply of care by family members and friends."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.utils import table

SOEP_IS_FATHER = 1
SOEP_IS_MOTHER = 2


def task_estimate_exogenous_informal_care_supply(
    path_to_sample: Path = BLD / "data" / "exog_care_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "exogenous_care_supply_transition_matrix.csv",
):

    df = pd.read_pickle(path_to_sample)
    df = df[df["female"] == 1].copy()

    # Create subsamples for mother and father
    # est_sample = df[df["female"] == 1]
    est_sample_mothers = df[df["ip03"] == SOEP_IS_MOTHER]

    # First, drop any rows with missing values in the relevant variables
    reg_data = est_sample_mothers[
        ["other_informal_care", "age", "has_sister", "education"]
    ].dropna()
    reg_data["age_squared"] = reg_data["age"] ** 2

    # Run logistic regression
    model = smf.logit(
        "other_informal_care ~ age + age_squared + has_sister + education",
        data=reg_data,
    ).fit()
    print(model.summary())

    # Create prediction grid
    ages = np.arange(30, 70)
    has_sister_vals = [0, 1]
    education_vals = [0, 1]

    rows = []

    for age in ages:
        for has_sister in has_sister_vals:
            for education in education_vals:
                age_squared = age**2
                pred_df = pd.DataFrame(
                    [
                        {
                            "age": age,
                            "age_squared": age_squared,
                            "has_sister": has_sister,
                            "education": education,
                        }
                    ]
                )
                prob = model.predict(pred_df)[0]

                rows.append(
                    {
                        "sex": 1,
                        "age": age,
                        "has_sister": has_sister,
                        "education": education,
                        "exog_care_prob": prob,
                    }
                )

    # Store as DataFrame
    exog_care_matrix = pd.DataFrame(rows)

    exog_care_matrix.to_csv(path_to_save, index=False)


def task_plot_exog_care_prob(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_exog_care_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "exogenous_care_supply_transition_matrix.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_exog_care_prob.png",
):
    """
    Plot predicted exogenous care probabilities by age,
    with lines for each (education, has_sister) combination.

    Parameters:
    - exog_care_matrix: DataFrame with columns:
        'sex', 'age', 'has_sister', 'education', 'exog_care_prob'
    """

    exog_care_matrix = pd.read_csv(path_to_exog_care_matrix)

    plt.figure(figsize=(10, 6))

    # Loop through the 4 combinations of education and has_sister
    for has_sister in (0, 1):
        for education in (0, 1):
            subset = exog_care_matrix[
                (exog_care_matrix["has_sister"] == has_sister)
                & (exog_care_matrix["education"] == education)
            ]
            label = f"has_sister={has_sister}, education={education}"
            plt.plot(subset["age"], subset["exog_care_prob"], label=label)

    plt.xlabel("Age")
    plt.ylabel("Predicted Probability of Other Informal Care")
    plt.title("Exogenous Informal Care Probability by Age")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path_to_save_plot)
    plt.close()


def task_plot_exog_care_prob_with_raw_data(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_sample: Path = BLD / "data" / "exog_care_estimation_sample.pkl",
    path_to_exog_care_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "exogenous_care_supply_transition_matrix.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_exog_care_prob_with_raw_data.png",
) -> None:
    """
    Plot predicted *and* observed exogenous care-supply probabilities.

    Parameters
    ----------
    path_to_exog_care_matrix
        CSV with columns ['sex','age','has_sister','education','exog_care_prob'].
    path_to_raw_sample
        Same sample you used for the estimation.
        Must contain variables
            • gender   (1 = Men, 2 = Women)
            • age
            • has_sister   (0 / 1)
            • education    (0 = low, 1 = high)
            • exog_care    (0 / 1)  … indicator whether care was actually supplied
    """

    # ─────────────────────────────────────────────────────────────────────
    # 1.  Load data
    # ─────────────────────────────────────────────────────────────────────
    df_pred = pd.read_csv(path_to_exog_care_matrix)

    df_raw = pd.read_pickle(path_to_raw_sample)
    df_raw = df_raw[df_raw["female"] == 1].copy()  # mothers only

    # ─────────────────────────────────────────────────────────────────────
    # 2.  Empirical frequencies
    # ─────────────────────────────────────────────────────────────────────
    df_emp = (
        df_raw.groupby(["age", "has_sister", "education"])["other_informal_care"]
        .agg(n_obs="count", n_care="sum")
        .reset_index()
    )
    df_emp["prob"] = df_emp["n_care"] / df_emp["n_obs"]

    # ─────────────────────────────────────────────────────────────────────
    # 3.  Plot
    # ─────────────────────────────────────────────────────────────────────
    colour_map = {
        (0, 0): "tab:blue",  # no sister, low edu
        (0, 1): "tab:green",  # no sister, high edu
        (1, 0): "tab:orange",  # sister,   low edu
        (1, 1): "tab:red",  # sister,   high edu
    }
    label_map = {
        (0, 0): "no sister / low edu",
        (0, 1): "no sister / high edu",
        (1, 0): "sister / low edu",
        (1, 1): "sister / high edu",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for sister in (0, 1):
        for edu in (0, 1):
            colour = colour_map[(sister, edu)]
            label = label_map[(sister, edu)]

            # predicted line
            dat_p = df_pred[
                (df_pred["has_sister"] == sister) & (df_pred["education"] == edu)
            ]
            ax.plot(dat_p["age"], dat_p["exog_care_prob"], color=colour, label=label)

            # empirical dots
            dat_e = df_emp[
                (df_emp["has_sister"] == sister) & (df_emp["education"] == edu)
            ]
            ax.scatter(dat_e["age"], dat_e["prob"], s=20, color=colour)

    ax.set_xlabel("Age")
    ax.set_ylabel("Probability of receiving other informal care")
    ax.set_title("Exogenous care supply: predicted vs. observed")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plt.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)
