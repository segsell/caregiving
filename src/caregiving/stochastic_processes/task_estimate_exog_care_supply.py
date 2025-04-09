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

SOEP_IS_FATHER = 1
SOEP_IS_MOTHER = 2


def task_estimate_exogenous_informal_care_supply(
    path_to_sample: Path = BLD / "data" / "exog_care_estimation_sample.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "exogenous_care_supply_transition_matrix.csv",
):

    data = pd.read_pickle(path_to_sample)
    data = data[data["female"] == 1].copy()

    # Create subsamples for mother and father
    # est_sample = df[df["female"] == 1]
    est_sample_mothers = data[data["ip03"] == SOEP_IS_MOTHER]

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
    ages = np.arange(40, 70)
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
