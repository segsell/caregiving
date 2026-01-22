"""Estimate formal care costs using OLS regression."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.shared import FEMALE, MAX_AGE_SIM
from caregiving.specs.task_write_specs import read_and_derive_specs


@pytask.mark.formal_care_costs
def task_estimate_formal_care_costs(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "formal_care_costs_sample.pkl",
    path_to_save_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "formal_care_costs_params.csv",
) -> None:
    """Estimate formal care costs using OLS regression.

    Regression specification:
        formal_care_costs ~ age + age^2 + education

    Only women are included in the estimation.

    """
    df = pd.read_pickle(path_to_data)

    # Filter to women only
    # df = df[df["sex"] == FEMALE].copy()

    # Filter to age <= MAX_AGE_SIM
    df = df[df["age"] <= MAX_AGE_SIM].copy()

    # Create age squared variable
    df["age_sq"] = df["age"] ** 2

    # Drop missing values for required variables
    required_vars = ["formal_care_costs", "age", "age_sq", "education"]
    df = df.dropna(subset=required_vars)

    print("\n" + "=" * 70)
    print("FORMAL CARE COSTS REGRESSION (Women Only)")
    print("=" * 70)
    print(f"Sample size: {len(df)}")
    print(f"Mean formal_care_costs: {df['formal_care_costs'].mean():.2f}")
    print(f"Std formal_care_costs: {df['formal_care_costs'].std():.2f}")
    print(f"\n{'=' * 70}\n")

    # Prepare regression variables
    exog_vars = ["age", "age_sq", "education"]
    X = sm.add_constant(df[exog_vars])
    y = df["formal_care_costs"]

    # Estimate OLS model
    model = sm.OLS(endog=y, exog=X)
    results = model.fit()

    # Create DataFrame with parameters
    params = pd.DataFrame(index=["coefficient"], columns=["const"] + exog_vars + ["N"])
    params.loc["coefficient", "const"] = results.params["const"]
    for var in exog_vars:
        params.loc["coefficient", var] = results.params[var]
    params.loc["coefficient", "N"] = len(df)

    # Save parameters
    params.to_csv(path_to_save_params)

    # Print summary
    print("OLS Regression Results:")
    print(f"  R-squared: {results.rsquared:.4f}")
    print(f"  Adjusted R-squared: {results.rsquared_adj:.4f}")
    print("\n  Parameters:")
    for param_name, param_value in results.params.items():
        print(f"    {param_name:15s}: {param_value:10.4f}")
    print(f"\n  Sample size: {len(df)}")
    print("=" * 70 + "\n")


@pytask.mark.formal_care_costs
def task_estimate_formal_care_costs_pooled(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "formal_care_costs_sample.pkl",
    path_to_save_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "formal_care_costs_params_pooled.csv",
) -> None:
    """Estimate formal care costs using OLS regression (pooled education).

    Regression specification:
        formal_care_costs ~ age + age^2

    No education distinction - pooled across all education levels.

    """
    df = pd.read_pickle(path_to_data)

    # Filter to age <= MAX_AGE_SIM
    df = df[df["age"] <= MAX_AGE_SIM].copy()

    # Create age squared variable
    df["age_sq"] = df["age"] ** 2

    # Drop missing values for required variables
    required_vars = ["formal_care_costs", "age", "age_sq"]
    df = df.dropna(subset=required_vars)

    print("\n" + "=" * 70)
    print("FORMAL CARE COSTS REGRESSION (Pooled Education)")
    print("=" * 70)
    print(f"Sample size: {len(df)}")
    print(f"Mean formal_care_costs: {df['formal_care_costs'].mean():.2f}")
    print(f"Std formal_care_costs: {df['formal_care_costs'].std():.2f}")
    print(f"\n{'=' * 70}\n")

    # Prepare regression variables (no education)
    exog_vars = ["age", "age_sq"]
    X = sm.add_constant(df[exog_vars])
    y = df["formal_care_costs"]

    # Estimate OLS model
    model = sm.OLS(endog=y, exog=X)
    results = model.fit()

    # Create DataFrame with parameters
    params = pd.DataFrame(index=["coefficient"], columns=["const"] + exog_vars + ["N"])
    params.loc["coefficient", "const"] = results.params["const"]
    for var in exog_vars:
        params.loc["coefficient", var] = results.params[var]
    params.loc["coefficient", "N"] = len(df)

    # Save parameters
    params.to_csv(path_to_save_params)

    # Print summary
    print("OLS Regression Results (Pooled):")
    print(f"  R-squared: {results.rsquared:.4f}")
    print(f"  Adjusted R-squared: {results.rsquared_adj:.4f}")
    print("\n  Parameters:")
    for param_name, param_value in results.params.items():
        print(f"    {param_name:15s}: {param_value:10.4f}")
    print(f"\n  Sample size: {len(df)}")
    print("=" * 70 + "\n")


@pytask.mark.formal_care_costs
def task_plot_formal_care_costs(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "formal_care_costs_params.csv",
    path_to_data: Path = BLD / "data" / "formal_care_costs_sample.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_formal_care_costs.png",
) -> None:
    """Plot predicted formal care costs by age for low and high education.

    Uses the estimated OLS parameters to predict formal care costs across ages
    for both education levels. Also plots raw data means.

    """
    specs = read_and_derive_specs(path_to_specs)
    params_df = pd.read_csv(path_to_params, index_col=0)

    # Load raw data
    df_raw = pd.read_pickle(path_to_data)
    # Filter to women only (if index has pid/syear, reset first)
    if isinstance(df_raw.index, pd.MultiIndex):
        df_raw = df_raw.reset_index()
    # df_raw = df_raw[df_raw["sex"] == FEMALE].copy()

    # Filter to age <= MAX_AGE_SIM
    df_raw = df_raw[df_raw["age"] <= MAX_AGE_SIM].copy()

    # Compute empirical means by age and education
    df_emp = (
        df_raw.groupby(["age", "education"])["formal_care_costs"]
        .agg(mean_cost="mean", n_obs="count")
        .reset_index()
    )

    # Extract parameters
    const = params_df.loc["coefficient", "const"]
    age_coef = params_df.loc["coefficient", "age"]
    age_sq_coef = params_df.loc["coefficient", "age_sq"]
    edu_coef = params_df.loc["coefficient", "education"]

    # Create age range (up to MAX_AGE_SIM inclusive)
    start_age = specs["start_age"]
    end_age = MAX_AGE_SIM
    ages = np.arange(start_age, end_age + 1)
    age_sq = ages**2

    # Get education labels
    edu_labels = specs["education_labels"]

    # Create predictions for each education level
    predictions = {}
    for edu_var, edu_label in enumerate(edu_labels):
        # Predict: const + age*age_coef + age_sq*age_sq_coef + education*edu_coef
        pred = const + age_coef * ages + age_sq_coef * age_sq + edu_coef * edu_var
        predictions[edu_label] = pred

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot predicted lines and raw data for each education level
    for edu_var, edu_label in enumerate(edu_labels):
        color = JET_COLOR_MAP[edu_var]

        # Plot predicted line
        ax.plot(
            ages,
            predictions[edu_label],
            label=edu_label,
            color=color,
            linewidth=2,
        )

        # Plot raw data means
        df_emp_edu = df_emp[df_emp["education"] == edu_var]
        ax.scatter(
            df_emp_edu["age"],
            df_emp_edu["mean_cost"],
            s=20,
            color=color,
            alpha=0.6,
            zorder=5,
        )

    # Set labels and limits
    ax.set_xlabel("Age")
    ax.set_ylabel("Formal Care Costs")
    ax.set_xlim(start_age, end_age)
    ax.legend(fontsize=9, title_fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)


@pytask.mark.formal_care_costs
def task_plot_formal_care_costs_pooled(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "formal_care_costs_params_pooled.csv",
    path_to_data: Path = BLD / "data" / "formal_care_costs_sample.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_formal_care_costs_pooled.png",
) -> None:
    """Plot predicted formal care costs by age (pooled education).

    Uses the estimated OLS parameters to predict formal care costs across ages.
    No education distinction - pooled across all education levels.
    Also plots raw data means.

    """
    specs = read_and_derive_specs(path_to_specs)
    params_df = pd.read_csv(path_to_params, index_col=0)

    # Load raw data
    df_raw = pd.read_pickle(path_to_data)
    # Filter to women only (if index has pid/syear, reset first)
    if isinstance(df_raw.index, pd.MultiIndex):
        df_raw = df_raw.reset_index()
    # df_raw = df_raw[df_raw["sex"] == FEMALE].copy()

    # Filter to age <= MAX_AGE_SIM
    df_raw = df_raw[df_raw["age"] <= MAX_AGE_SIM].copy()

    # Compute empirical means by age (pooled across education)
    df_emp = (
        df_raw.groupby("age")["formal_care_costs"]
        .agg(mean_cost="mean", n_obs="count")
        .reset_index()
    )

    # Extract parameters
    const = params_df.loc["coefficient", "const"]
    age_coef = params_df.loc["coefficient", "age"]
    age_sq_coef = params_df.loc["coefficient", "age_sq"]

    # Create age range (up to MAX_AGE_SIM inclusive)
    start_age = specs["start_age"]
    end_age = MAX_AGE_SIM
    ages = np.arange(start_age, end_age + 1)
    age_sq = ages**2

    # Create predictions (no education term)
    predictions = const + age_coef * ages + age_sq_coef * age_sq

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot predicted line
    ax.plot(
        ages,
        predictions,
        label="Predicted",
        color=JET_COLOR_MAP[0],
        linewidth=2,
    )

    # Plot raw data means
    ax.scatter(
        df_emp["age"],
        df_emp["mean_cost"],
        s=20,
        color=JET_COLOR_MAP[0],
        alpha=0.6,
        zorder=5,
        label="Observed mean",
    )

    # Set labels and limits
    ax.set_xlabel("Age")
    ax.set_ylabel("Formal Care Costs")
    ax.set_xlim(start_age, end_age)
    ax.legend(fontsize=9, title_fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)
