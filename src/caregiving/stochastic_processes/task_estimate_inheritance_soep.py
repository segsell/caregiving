"""Estimate inheritance probability and amount regressions."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs

# Minimum sample size for regression estimation
MIN_SAMPLE_SIZE = 50


@pytask.mark.inheritance
def task_estimate_inheritance(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_logit_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_logit_params.csv",
    path_to_save_amount_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_params.csv",
) -> None:
    """Estimate inheritance probability (logit) and amount (OLS) by sex.

    Regressions condition on parental death in t or t-1.

    Logit regression:
        P(inheritance_this_year=1) ~ age + age^2 + lagged_light_care +
                                     lagged_intensive_care + education

    OLS regression (on ln(inheritance_amount)):
        ln(inheritance_amount) ~ age + age^2 + lagged_light_care +
                                 lagged_intensive_care + education

    """
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=0)

    # Set values above 90th percentile to NaN
    p90_threshold = df["inheritance_amount"].quantile(0.90)
    df.loc[df["inheritance_amount"] > p90_threshold, "inheritance_amount"] = np.nan

    # # Drop observations above 90th percentile
    # df = df[df["inheritance_amount"] <= p90_threshold]

    # Prepare data
    df = prepare_inheritance_data(df)

    # Estimate logit for probability of positive inheritance
    logit_params = estimate_inheritance_logit_by_sex(df, specs)

    # Estimate OLS for ln(inheritance_amount) conditional on positive inheritance
    amount_params = estimate_inheritance_amount_by_sex(df, specs)

    # Save results
    logit_params.to_csv(path_to_save_logit_params)
    amount_params.to_csv(path_to_save_amount_params)

    # Print summary
    print("\n" + "=" * 70)
    print("INHERITANCE REGRESSION RESULTS")
    print("=" * 70)
    print("\nLogit Regression Parameters (Probability of Positive Inheritance):")
    print(logit_params)
    print("\nOLS Regression Parameters (ln(Inheritance Amount)):")
    print(amount_params)
    print("=" * 70 + "\n")


def prepare_inheritance_data(df):
    """Prepare data for inheritance regressions."""
    # Create squared age term
    df["age_sq"] = df["age"] ** 2

    # Create lagged care variables
    # Note: pid is now a column, not an index
    df = df.sort_values(["pid", "syear"])
    df["lagged_light_care"] = df.groupby("pid")["light_care"].shift(1)
    df["lagged_intensive_care"] = df.groupby("pid")["intensive_care"].shift(1)

    # Create indicators for recent caregiving (current or previous period)
    # Light care in t or t-1
    df["light_care_recent"] = (
        (df["light_care"] > 0) | (df["lagged_light_care"] > 0)
    ).astype(int)

    # Intensive care in t or t-1
    df["intensive_care_recent"] = (
        (df["intensive_care"] > 0) | (df["lagged_intensive_care"] > 0)
    ).astype(int)

    # Any care (light or intensive) in t or t-1
    df["any_care_recent"] = (
        (df["any_care"] > 0) | (df.groupby("pid")["any_care"].shift(1) > 0)
    ).astype(int)

    # Create parent death indicator for t or t-1
    df["lagged_mother_died"] = df.groupby("pid")["mother_died_this_year"].shift(1)
    df["lagged_father_died"] = df.groupby("pid")["father_died_this_year"].shift(1)
    df["lagged_parent_died"] = (
        (df["lagged_mother_died"] == 1) | (df["lagged_father_died"] == 1)
    ).astype(int)
    df["parent_died_this_year"] = (
        (df["mother_died_this_year"] == 1) | (df["father_died_this_year"] == 1)
    ).astype(int)

    # Parent died in t or t-1
    df["parent_died_recent"] = (
        (df["mother_died_this_year"] == 1)
        | (df["lagged_mother_died"] == 1)
        | (df["father_died_this_year"] == 1)
        | (df["lagged_father_died"] == 1)
    ).astype(int)

    # # Create spouse death indicator
    # # Spouse died this year if: had partner in t-1 (partner_state > 0)
    # # and no partner in t (partner_state == 0)
    # df["lagged_partner_state"] = df.groupby("pid")["partner_state"].shift(1)
    # df["spouse_died_this_year"] = (
    #     (df["lagged_partner_state"] > 0) & (df["partner_state"] == 0)
    # ).astype(int)

    # # Spouse died in t or t-1
    # df["lagged_spouse_died"] = df.groupby("pid")["spouse_died_this_year"].shift(1)
    # df["spouse_died_recent"] = (
    #     (df["spouse_died_this_year"] == 1) | (df["lagged_spouse_died"] == 1)
    # ).astype(int)

    # Create ln(inheritance_amount) for OLS regression
    # Only for observations with positive inheritance amount
    df["ln_inheritance_amount"] = np.nan
    positive_inheritance = df["inheritance_amount"].notna() & (
        df["inheritance_amount"] > 0
    )

    df.loc[positive_inheritance, "ln_inheritance_amount"] = np.log(
        df.loc[positive_inheritance, "inheritance_amount"]
    )
    # df.loc[positive_inheritance, "ln_inheritance_amount"] = df.loc[
    #     positive_inheritance, "inheritance_amount"
    # ]

    return df


def estimate_inheritance_logit_by_sex(df, specs):
    """Estimate logit model for probability of positive inheritance by sex.

    No filtering - uses all observations.

    Specification:
        P(inheritance_this_year=1) ~ age + age² + any_care_recent +
                                     parent_died_recent + education

    The parent_died_recent variable captures parental death in t or t-1.
    """
    index = pd.Index(specs["sex_labels"], name="sex")
    columns = [
        "age",
        "age_sq",
        "any_care_recent",
        "parent_died_recent",
        "education",
        "const",
    ]
    logit_params = pd.DataFrame(index=index, columns=columns)

    # Use all observations (no filtering)
    df_filtered = df.copy()

    print("\n" + "=" * 70)
    print("LOGIT REGRESSION: Probability of Positive Inheritance")
    print("=" * 70)
    print(f"Total observations: {len(df_filtered)}")

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()

        # Drop observations with missing values in any regressor
        df_sex = df_sex.dropna(
            subset=[
                "inheritance_this_year",
                "age",
                "age_sq",
                "any_care_recent",
                "parent_died_recent",
                "education",
            ]
        )

        print(f"\n{sex_label}:")
        print(f"  Observations: {len(df_sex)}")
        print(f"  Inheritance rate: {df_sex['inheritance_this_year'].mean():.3f}")

        if len(df_sex) < MIN_SAMPLE_SIZE:
            print(
                f"  WARNING: Too few observations for {sex_label}, "
                f"skipping regression"
            )
            continue

        # Estimate logit model
        exog_vars = [
            "age",
            "age_sq",
            "any_care_recent",
            "parent_died_recent",
            "education",
        ]
        X = sm.add_constant(df_sex[exog_vars])
        y = df_sex["inheritance_this_year"]

        try:
            model = sm.Logit(endog=y, exog=X)
            results = model.fit(disp=False)

            # Save parameters
            logit_params.loc[sex_label, "const"] = results.params["const"]
            for var in exog_vars:
                logit_params.loc[sex_label, var] = results.params[var]

            # Print summary
            print(f"  Model converged: {results.mle_retvals['converged']}")
            print(f"  Pseudo R-squared: {results.prsquared:.4f}")
            print("  Parameters:")
            for param_name, param_value in results.params.items():
                print(f"    {param_name:25s}: {param_value:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return logit_params


def estimate_inheritance_amount_by_sex(df, specs):
    """Estimate OLS model for ln(inheritance_amount) by sex.

    Conditions on parent death in t or t-1 and positive inheritance.
    """
    index = pd.Index(specs["sex_labels"], name="sex")
    columns = [
        "age",
        "age_sq",
        "light_care_recent",
        "intensive_care_recent",
        "education",
        "const",
    ]
    amount_params = pd.DataFrame(index=index, columns=columns)

    # Filter to observations where parent died in t or t-1 AND positive inheritance
    df_filtered = df[
        (df["parent_died_recent"] == 1) & (df["ln_inheritance_amount"].notna())
    ].copy()

    print("\n" + "=" * 70)
    print("OLS REGRESSION: ln(Inheritance Amount)")
    print("=" * 70)
    print(
        f"Total observations after filtering for parent death & "
        f"positive inheritance: {len(df_filtered)}"
    )

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()

        # Drop observations with missing values in any regressor
        df_sex = df_sex.dropna(
            subset=[
                "ln_inheritance_amount",
                "age",
                "age_sq",
                "light_care_recent",
                "intensive_care_recent",
                "education",
            ]
        )

        print(f"\n{sex_label}:")
        print(f"  Observations: {len(df_sex)}")
        if len(df_sex) > 0:
            print(
                f"  Mean ln(inheritance): {df_sex['ln_inheritance_amount'].mean():.3f}"
            )
            mean_amount = np.exp(df_sex["ln_inheritance_amount"]).mean()
            print(f"  Mean inheritance (€): {mean_amount:,.2f}")

        if len(df_sex) < MIN_SAMPLE_SIZE:
            print(
                f"  WARNING: Too few observations for {sex_label}, "
                f"skipping regression"
            )
            continue

        # Estimate OLS model
        exog_vars = [
            "age",
            "age_sq",
            "light_care_recent",
            "intensive_care_recent",
            "education",
        ]
        X = sm.add_constant(df_sex[exog_vars])
        y = df_sex["ln_inheritance_amount"]

        try:
            model = sm.OLS(endog=y, exog=X)
            results = model.fit()

            # Save parameters
            amount_params.loc[sex_label, "const"] = results.params["const"]
            for var in exog_vars:
                amount_params.loc[sex_label, var] = results.params[var]

            # Print summary
            print(f"  R-squared: {results.rsquared:.4f}")
            print("  Parameters:")
            for param_name, param_value in results.params.items():
                print(f"    {param_name:25s}: {param_value:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return amount_params
