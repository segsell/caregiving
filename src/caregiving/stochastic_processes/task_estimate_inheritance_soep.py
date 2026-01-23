"""Estimate inheritance probability and amount regressions."""

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
INHERITANCE_QUANTILE_THRESHOLD = 0.90


def deflate_inheritance_amount_for_estimation(df, cpi_data, specs):
    """Deflate inheritance amount using consumer price index.

    Similar to _deflate_inheritance_amount but works with DataFrame
    that may not have MultiIndex (pid, syear).

    Args:
        df: DataFrame containing inheritance_amount and year_inheritance columns
        cpi_data: DataFrame with CPI data (should have int_year and cpi columns)
        specs: Dictionary with specs including reference_year

    Returns:
        DataFrame with deflated inheritance_amount
    """
    # Prepare CPI data
    cpi_data_copy = cpi_data.rename(columns={"int_year": "year_inheritance"})

    _base_year = specs["reference_year"]
    base_year_cpi = cpi_data_copy.loc[
        cpi_data_copy["year_inheritance"] == _base_year, "cpi"
    ].iloc[0]

    cpi_data_copy["cpi_normalized"] = cpi_data_copy["cpi"] / base_year_cpi

    # Merge CPI data on year_inheritance
    df = df.merge(
        cpi_data_copy[["year_inheritance", "cpi_normalized"]],
        on="year_inheritance",
        how="left",
    )

    # Deflate inheritance amount (only where both are not NaN)
    mask = df["inheritance_amount"].notna() & df["cpi_normalized"].notna()
    df.loc[mask, "inheritance_amount"] = (
        df.loc[mask, "inheritance_amount"] / df.loc[mask, "cpi_normalized"]
    )

    # Drop temporary CPI column
    df = df.drop(columns=["cpi_normalized"])

    return df


@pytask.mark.inheritance
def task_estimate_inheritance_specifications(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_specs"
    / "_specs_summary.txt",
) -> None:
    """Estimate 12 different specifications for inheritance probability.

    CPI deflation is applied to inheritance_amount before estimation.

    Tests different combinations of care and parent death timing variables.
    """
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=0)

    # Load CPI data and deflate inheritance_amount
    cpi_data = pd.read_csv(path_to_cpi, index_col=0)
    df = deflate_inheritance_amount_for_estimation(df, cpi_data, specs)

    # Set values above 90th percentile to NaN
    p90_threshold = df["inheritance_amount"].quantile(INHERITANCE_QUANTILE_THRESHOLD)
    df.loc[df["inheritance_amount"] > p90_threshold, "inheritance_amount"] = np.nan

    # Prepare data
    df = prepare_inheritance_data(df)

    # Create output directory
    path_to_save_dir = path_to_save_summary.parent
    path_to_save_dir.mkdir(parents=True, exist_ok=True)

    # Define all 12 specifications
    specifications = [
        # Spec 1-3: No filtering, different timing combinations
        {
            "name": "spec1_any_care_parent_this_year",
            "care_var": "any_care",
            "parent_var": "parent_died_this_year",
            "filter": None,
        },
        {
            "name": "spec2_any_care_recent_parent_recent",
            "care_var": "any_care_recent",
            "parent_var": "parent_died_recent",
            "filter": None,
        },
        {
            "name": "spec3_any_care_last_year_parent_last_year",
            "care_var": "any_care_last_year",
            "parent_var": "parent_died_last_year",
            "filter": None,
        },
        # Spec 4-6: Filter on parent_died_this_year
        {
            "name": "spec4_any_care_last_year_filter_parent_this_year",
            "care_var": "any_care_last_year",
            "parent_var": None,  # Not included as variable, only as filter
            "filter": "parent_died_this_year == 1",
        },
        # Spec 7-9: Different care timing with parent_died_this_year filter
        {
            "name": "spec7_any_care_this_year_filter_parent_this_year",
            "care_var": "any_care",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec10_any_care_recent_filter_parent_this_year",
            "care_var": "any_care_recent",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        # Spec 5,8,11: Filter on parent_died_last_year
        {
            "name": "spec5_any_care_last_year_filter_parent_last_year",
            "care_var": "any_care_last_year",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec8_any_care_this_year_filter_parent_last_year",
            "care_var": "any_care",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec11_any_care_recent_filter_parent_last_year",
            "care_var": "any_care_recent",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        # Spec 6,9,12: Filter on parent_died_recent
        {
            "name": "spec6_any_care_last_year_filter_parent_recent",
            "care_var": "any_care_last_year",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec9_any_care_this_year_filter_parent_recent",
            "care_var": "any_care",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec12_any_care_recent_filter_parent_recent",
            "care_var": "any_care_recent",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
    ]

    # Run all specifications
    results = {}
    for spec in specifications:
        params = estimate_logit_specification(
            df=df,
            specs=specs,
            care_var=spec["care_var"],
            parent_var=spec["parent_var"],
            filter_condition=spec["filter"],
            spec_name=spec["name"],
        )

        # Save results
        output_path = path_to_save_dir / f"{spec['name']}_params.csv"
        params.to_csv(output_path)
        results[spec["name"]] = params

    # Save summary
    print("\n" + "=" * 70)
    print("ALL SPECIFICATIONS COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {path_to_save_dir}")

    # Write summary file for pytask dependency tracking
    with path_to_save_summary.open("w") as f:
        f.write("Inheritance Probability Specifications Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total specifications: {len(specifications)}\n")
        f.write(f"Output directory: {path_to_save_dir}\n\n")
        f.write("Specifications:\n")
        for spec in specifications:
            f.write(f"  - {spec['name']}\n")
            f.write(f"    Care: {spec['care_var']}\n")
            f.write(f"    Parent: {spec['parent_var']}\n")
            f.write(f"    Filter: {spec['filter']}\n\n")


@pytask.mark.inheritance
def task_estimate_inheritance(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
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

    CPI deflation is applied to inheritance_amount before estimation.

    Logit regression:
        P(inheritance_this_year=1) ~ age + age^2 + lagged_light_care +
                                     lagged_intensive_care + education

    OLS regression (on ln(inheritance_amount)):
        ln(inheritance_amount) ~ age + age^2 + lagged_light_care +
                                 lagged_intensive_care + education

    """
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=0)

    # Load CPI data and deflate inheritance_amount
    cpi_data = pd.read_csv(path_to_cpi, index_col=0)
    df = deflate_inheritance_amount_for_estimation(df, cpi_data, specs)

    # Set values above 90th percentile to NaN
    p90_threshold = df["inheritance_amount"].quantile(INHERITANCE_QUANTILE_THRESHOLD)
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

    # Create caregiving indicators for different time periods
    # Previous period (last year) - lagged
    df["lagged_any_care"] = df.groupby("pid")["any_care"].shift(1)
    df["any_care_last_year"] = (df["lagged_any_care"] > 0).astype(int)
    df["light_care_last_year"] = (df["lagged_light_care"] > 0).astype(int)
    df["intensive_care_last_year"] = (df["lagged_intensive_care"] > 0).astype(int)

    # Recent period (t or t-1)
    df["light_care_recent"] = (
        (df["light_care"] > 0) | (df["lagged_light_care"] > 0)
    ).astype(int)

    df["intensive_care_recent"] = (
        (df["intensive_care"] > 0) | (df["lagged_intensive_care"] > 0)
    ).astype(int)

    df["any_care_recent"] = ((df["any_care"] > 0) | (df["lagged_any_care"] > 0)).astype(
        int
    )

    # Create parent death indicators for different time periods
    df["lagged_mother_died"] = df.groupby("pid")["mother_died_this_year"].shift(1)
    df["lagged_father_died"] = df.groupby("pid")["father_died_this_year"].shift(1)

    # Current period (this year)
    df["parent_died_this_year"] = (
        (df["mother_died_this_year"] == 1) | (df["father_died_this_year"] == 1)
    ).astype(int)

    # Previous period (last year)
    df["parent_died_last_year"] = (
        (df["lagged_mother_died"] == 1) | (df["lagged_father_died"] == 1)
    ).astype(int)

    # Recent period (t or t-1)
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
        "N",
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
            # Save sample size
            logit_params.loc[sex_label, "N"] = len(df_sex)

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
        "N",
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
            # Save sample size
            amount_params.loc[sex_label, "N"] = len(df_sex)

            # Print summary
            print(f"  R-squared: {results.rsquared:.4f}")
            print("  Parameters:")
            for param_name, param_value in results.params.items():
                print(f"    {param_name:25s}: {param_value:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return amount_params


def estimate_logit_specification(
    df, specs, care_var, parent_var=None, filter_condition=None, spec_name="spec"
):
    """Estimate a single logit specification with flexible variables.

    Args:
        df: DataFrame with prepared data
        specs: Model specifications
        care_var: Name of caregiving variable to include
        parent_var: Name of parent death variable to include (None if using filter only)
        filter_condition: Optional filtering condition
        spec_name: Name for this specification (for output)

    Returns:
        DataFrame with estimated parameters (includes N = sample size)
    """
    index = pd.Index(specs["sex_labels"], name="sex")

    # Build columns list based on whether parent_var is included
    if parent_var is not None:
        columns = ["age", "age_sq", care_var, parent_var, "education", "const", "N"]
        exog_vars_base = ["age", "age_sq", care_var, parent_var, "education"]
    else:
        columns = ["age", "age_sq", care_var, "education", "const", "N"]
        exog_vars_base = ["age", "age_sq", care_var, "education"]

    params = pd.DataFrame(index=index, columns=columns)

    # Apply filter if specified
    if filter_condition is not None:
        df_filtered = df.query(filter_condition).copy()
        filter_desc = filter_condition
    else:
        df_filtered = df.copy()
        filter_desc = "None"

    print("\n" + "=" * 70)
    print(f"SPECIFICATION: {spec_name}")
    print(f"Care variable: {care_var}")
    print(f"Parent variable: {parent_var or 'N/A (filtered only)'}")
    print(f"Filter: {filter_desc}")
    print("=" * 70)
    print(f"Total observations: {len(df_filtered)}")

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()

        # Drop missing values
        required_vars = [
            "inheritance_this_year",
            "age",
            "age_sq",
            care_var,
            "education",
        ]
        if parent_var is not None:
            required_vars.append(parent_var)
        df_sex = df_sex.dropna(subset=required_vars)

        print(f"\n{sex_label}:")
        print(f"  Observations: {len(df_sex)}")
        print(f"  Inheritance rate: {df_sex['inheritance_this_year'].mean():.3f}")

        if len(df_sex) < MIN_SAMPLE_SIZE:
            print("  WARNING: Too few observations, skipping")
            continue

        # Estimate logit model
        X = sm.add_constant(df_sex[exog_vars_base])
        y = df_sex["inheritance_this_year"]

        try:
            model = sm.Logit(endog=y, exog=X)
            results = model.fit(disp=False)

            # Save parameters
            params.loc[sex_label, "const"] = results.params["const"]
            for var in exog_vars_base:
                params.loc[sex_label, var] = results.params[var]
            # Save sample size
            params.loc[sex_label, "N"] = len(df_sex)

            # Print summary
            print(f"  Model converged: {results.mle_retvals['converged']}")
            print(f"  Pseudo R-squared: {results.prsquared:.4f}")
            print("  Key parameters:")
            print(f"    {care_var:25s}: {results.params[care_var]:8.4f}")
            if parent_var is not None:
                print(f"    {parent_var:25s}: {results.params[parent_var]:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return params


def estimate_ols_specification(
    df, specs, care_var, parent_var=None, filter_condition=None, spec_name="spec"
):
    """Estimate a single OLS specification for ln(inheritance_amount).

    Args:
        df: DataFrame with prepared data
        specs: Model specifications
        care_var: Name of caregiving variable to include
        parent_var: Name of parent death variable to include (None if using filter only)
        filter_condition: Optional filtering condition
        spec_name: Name for this specification (for output)

    Returns:
        DataFrame with estimated parameters (includes N = sample size)
    """
    index = pd.Index(specs["sex_labels"], name="sex")

    # Build columns list based on whether parent_var is included
    if parent_var is not None:
        columns = ["age", "age_sq", care_var, parent_var, "education", "const", "N"]
        exog_vars_base = ["age", "age_sq", care_var, parent_var, "education"]
    else:
        columns = ["age", "age_sq", care_var, "education", "const", "N"]
        exog_vars_base = ["age", "age_sq", care_var, "education"]

    params = pd.DataFrame(index=index, columns=columns)

    # Apply filter if specified
    if filter_condition is not None:
        df_filtered = df.query(filter_condition).copy()
        filter_desc = filter_condition
    else:
        df_filtered = df.copy()
        filter_desc = "None"

    # Additional filter: only positive inheritance amounts
    df_filtered = df_filtered[df_filtered["ln_inheritance_amount"].notna()].copy()

    print("\n" + "=" * 70)
    print(f"SPECIFICATION: {spec_name}")
    print(f"Care variable: {care_var}")
    print(f"Parent variable: {parent_var or 'N/A (filtered only)'}")
    print(f"Filter: {filter_desc}")
    print("=" * 70)
    print(f"Total observations: {len(df_filtered)}")

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()

        # Drop missing values
        required_vars = [
            "ln_inheritance_amount",
            "age",
            "age_sq",
            care_var,
            "education",
        ]
        if parent_var is not None:
            required_vars.append(parent_var)
        df_sex = df_sex.dropna(subset=required_vars)

        print(f"\n{sex_label}:")
        print(f"  Observations: {len(df_sex)}")
        if len(df_sex) > 0:
            print(
                f"  Mean ln(inheritance): {df_sex['ln_inheritance_amount'].mean():.3f}"
            )

        if len(df_sex) < MIN_SAMPLE_SIZE:
            print("  WARNING: Too few observations, skipping")
            continue

        # Estimate OLS model
        X = sm.add_constant(df_sex[exog_vars_base])
        y = df_sex["ln_inheritance_amount"]

        try:
            model = sm.OLS(endog=y, exog=X)
            results = model.fit()

            # Save parameters
            params.loc[sex_label, "const"] = results.params["const"]
            for var in exog_vars_base:
                params.loc[sex_label, var] = results.params[var]
            # Save sample size
            params.loc[sex_label, "N"] = len(df_sex)

            # Print summary
            print(f"  R-squared: {results.rsquared:.4f}")
            print("  Key parameters:")
            print(f"    {care_var:25s}: {results.params[care_var]:8.4f}")
            if parent_var is not None:
                print(f"    {parent_var:25s}: {results.params[parent_var]:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return params


def estimate_logit_specification_two_care(  # noqa: PLR0915
    df,
    specs,
    light_care_var,
    intensive_care_var,
    parent_var=None,
    filter_condition=None,
    spec_name="spec",
):
    """Estimate logit with separate light and intensive care variables.

    Args:
        df: DataFrame with prepared data
        specs: Model specifications
        light_care_var: Name of light caregiving variable
        intensive_care_var: Name of intensive caregiving variable
        parent_var: Name of parent death variable (None if using filter only)
        filter_condition: Optional filtering condition
        spec_name: Name for this specification

    Returns:
        DataFrame with estimated parameters (includes N)
    """
    index = pd.Index(specs["sex_labels"], name="sex")

    # Build columns list
    if parent_var is not None:
        columns = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            parent_var,
            "education",
            "const",
            "N",
        ]
        exog_vars_base = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            parent_var,
            "education",
        ]
    else:
        columns = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            "education",
            "const",
            "N",
        ]
        exog_vars_base = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            "education",
        ]

    params = pd.DataFrame(index=index, columns=columns)

    # Apply filter
    if filter_condition is not None:
        df_filtered = df.query(filter_condition).copy()
        filter_desc = filter_condition
    else:
        df_filtered = df.copy()
        filter_desc = "None"

    print("\n" + "=" * 70)
    print(f"SPECIFICATION: {spec_name}")
    print(f"Light care variable: {light_care_var}")
    print(f"Intensive care variable: {intensive_care_var}")
    print(f"Parent variable: {parent_var or 'N/A (filtered only)'}")
    print(f"Filter: {filter_desc}")
    print("=" * 70)
    print(f"Total observations: {len(df_filtered)}")

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()

        # Drop missing values
        required_vars = [
            "inheritance_this_year",
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            "education",
        ]
        if parent_var is not None:
            required_vars.append(parent_var)
        df_sex = df_sex.dropna(subset=required_vars)

        print(f"\n{sex_label}:")
        print(f"  Observations: {len(df_sex)}")
        print(f"  Inheritance rate: {df_sex['inheritance_this_year'].mean():.3f}")

        if len(df_sex) < MIN_SAMPLE_SIZE:
            print("  WARNING: Too few observations, skipping")
            continue

        # Estimate logit model
        X = sm.add_constant(df_sex[exog_vars_base])
        y = df_sex["inheritance_this_year"]

        try:
            model = sm.Logit(endog=y, exog=X)
            results = model.fit(disp=False)

            # Save parameters
            params.loc[sex_label, "const"] = results.params["const"]
            for var in exog_vars_base:
                params.loc[sex_label, var] = results.params[var]
            # Save sample size
            params.loc[sex_label, "N"] = len(df_sex)

            # Print summary
            print(f"  Model converged: {results.mle_retvals['converged']}")
            print(f"  Pseudo R-squared: {results.prsquared:.4f}")
            print("  Key parameters:")
            print(f"    {light_care_var:25s}: {results.params[light_care_var]:8.4f}")
            print(
                f"    {intensive_care_var:25s}: "
                f"{results.params[intensive_care_var]:8.4f}"
            )
            if parent_var is not None:
                print(f"    {parent_var:25s}: {results.params[parent_var]:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return params


def estimate_ols_specification_two_care(  # noqa: PLR0915
    df,
    specs,
    light_care_var,
    intensive_care_var,
    parent_var=None,
    filter_condition=None,
    spec_name="spec",
):
    """Estimate OLS with separate light and intensive care variables.

    Args:
        df: DataFrame with prepared data
        specs: Model specifications
        light_care_var: Name of light caregiving variable
        intensive_care_var: Name of intensive caregiving variable
        parent_var: Name of parent death variable (None if using filter only)
        filter_condition: Optional filtering condition
        spec_name: Name for this specification

    Returns:
        DataFrame with estimated parameters (includes N)
    """
    index = pd.Index(specs["sex_labels"], name="sex")

    # Build columns list
    if parent_var is not None:
        columns = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            parent_var,
            "education",
            "const",
            "N",
        ]
        exog_vars_base = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            parent_var,
            "education",
        ]
    else:
        columns = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            "education",
            "const",
            "N",
        ]
        exog_vars_base = [
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            "education",
        ]

    params = pd.DataFrame(index=index, columns=columns)

    # Apply filter
    if filter_condition is not None:
        df_filtered = df.query(filter_condition).copy()
        filter_desc = filter_condition
    else:
        df_filtered = df.copy()
        filter_desc = "None"

    # Additional filter: only positive inheritance
    df_filtered = df_filtered[df_filtered["ln_inheritance_amount"].notna()].copy()

    print("\n" + "=" * 70)
    print(f"SPECIFICATION: {spec_name}")
    print(f"Light care variable: {light_care_var}")
    print(f"Intensive care variable: {intensive_care_var}")
    print(f"Parent variable: {parent_var or 'N/A (filtered only)'}")
    print(f"Filter: {filter_desc}")
    print("=" * 70)
    print(f"Total observations: {len(df_filtered)}")

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()

        # Drop missing values
        required_vars = [
            "ln_inheritance_amount",
            "age",
            "age_sq",
            light_care_var,
            intensive_care_var,
            "education",
        ]
        if parent_var is not None:
            required_vars.append(parent_var)
        df_sex = df_sex.dropna(subset=required_vars)

        print(f"\n{sex_label}:")
        print(f"  Observations: {len(df_sex)}")
        if len(df_sex) > 0:
            print(
                f"  Mean ln(inheritance): {df_sex['ln_inheritance_amount'].mean():.3f}"
            )

        if len(df_sex) < MIN_SAMPLE_SIZE:
            print("  WARNING: Too few observations, skipping")
            continue

        # Estimate OLS model
        X = sm.add_constant(df_sex[exog_vars_base])
        y = df_sex["ln_inheritance_amount"]

        try:
            model = sm.OLS(endog=y, exog=X)
            results = model.fit()

            # Save parameters
            params.loc[sex_label, "const"] = results.params["const"]
            for var in exog_vars_base:
                params.loc[sex_label, var] = results.params[var]
            # Save sample size
            params.loc[sex_label, "N"] = len(df_sex)

            # Print summary
            print(f"  R-squared: {results.rsquared:.4f}")
            print("  Key parameters:")
            print(f"    {light_care_var:25s}: {results.params[light_care_var]:8.4f}")
            print(
                f"    {intensive_care_var:25s}: "
                f"{results.params[intensive_care_var]:8.4f}"
            )
            if parent_var is not None:
                print(f"    {parent_var:25s}: {results.params[parent_var]:8.4f}")

        except Exception as e:
            print(f"  ERROR: Model estimation failed: {e}")

    return params


@pytask.mark.inheritance
def task_estimate_inheritance_amount_specifications(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_specs"
    / "_specs_summary.txt",
) -> None:
    """Estimate 12 different specifications for ln(inheritance_amount).

    Tests different combinations of care and parent death timing variables.
    Same specifications as the logit model, but for amount conditional on
    positive inheritance.

    CPI deflation is applied to inheritance_amount before estimation.
    """
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=0)

    # Load CPI data and deflate inheritance_amount
    cpi_data = pd.read_csv(path_to_cpi, index_col=0)
    df = deflate_inheritance_amount_for_estimation(df, cpi_data, specs)

    # Set values above 90th percentile to NaN
    p90_threshold = df["inheritance_amount"].quantile(INHERITANCE_QUANTILE_THRESHOLD)
    df.loc[df["inheritance_amount"] > p90_threshold, "inheritance_amount"] = np.nan

    # Prepare data
    df = prepare_inheritance_data(df)

    # Create output directory
    path_to_save_dir = path_to_save_summary.parent
    path_to_save_dir.mkdir(parents=True, exist_ok=True)

    # Define all 12 specifications (same as logit)
    specifications = [
        # Spec 1-3: No filtering, different timing combinations
        {
            "name": "spec1_any_care_parent_this_year",
            "care_var": "any_care",
            "parent_var": "parent_died_this_year",
            "filter": None,
        },
        {
            "name": "spec2_any_care_recent_parent_recent",
            "care_var": "any_care_recent",
            "parent_var": "parent_died_recent",
            "filter": None,
        },
        {
            "name": "spec3_any_care_last_year_parent_last_year",
            "care_var": "any_care_last_year",
            "parent_var": "parent_died_last_year",
            "filter": None,
        },
        # Spec 4-6: Filter on parent_died_this_year
        {
            "name": "spec4_any_care_last_year_filter_parent_this_year",
            "care_var": "any_care_last_year",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec7_any_care_this_year_filter_parent_this_year",
            "care_var": "any_care",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec10_any_care_recent_filter_parent_this_year",
            "care_var": "any_care_recent",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        # Spec 5,8,11: Filter on parent_died_last_year
        {
            "name": "spec5_any_care_last_year_filter_parent_last_year",
            "care_var": "any_care_last_year",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec8_any_care_this_year_filter_parent_last_year",
            "care_var": "any_care",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec11_any_care_recent_filter_parent_last_year",
            "care_var": "any_care_recent",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        # Spec 6,9,12: Filter on parent_died_recent
        {
            "name": "spec6_any_care_last_year_filter_parent_recent",
            "care_var": "any_care_last_year",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec9_any_care_this_year_filter_parent_recent",
            "care_var": "any_care",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec12_any_care_recent_filter_parent_recent",
            "care_var": "any_care_recent",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
    ]

    # Run all specifications
    results = {}
    for spec in specifications:
        params = estimate_ols_specification(
            df=df,
            specs=specs,
            care_var=spec["care_var"],
            parent_var=spec["parent_var"],
            filter_condition=spec["filter"],
            spec_name=spec["name"],
        )

        # Save results
        output_path = path_to_save_dir / f"{spec['name']}_params.csv"
        params.to_csv(output_path)
        results[spec["name"]] = params

    # Save summary
    print("\n" + "=" * 70)
    print("ALL AMOUNT SPECIFICATIONS COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {path_to_save_dir}")

    # Write summary file for pytask dependency tracking
    with path_to_save_summary.open("w") as f:
        f.write("Inheritance Amount Specifications Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total specifications: {len(specifications)}\n")
        f.write(f"Output directory: {path_to_save_dir}\n\n")
        f.write("Specifications:\n")
        for spec in specifications:
            f.write(f"  - {spec['name']}\n")
            f.write(f"    Care: {spec['care_var']}\n")
            f.write(f"    Parent: {spec['parent_var']}\n")
            f.write(f"    Filter: {spec['filter']}\n\n")


@pytask.mark.inheritance
def task_estimate_inheritance_prob_two_care_specifications(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_specs_two_care"
    / "_specs_summary.txt",
) -> None:
    """Estimate 12 specifications with separate light/intensive care variables.

    For inheritance probability (logit).

    CPI deflation is applied to inheritance_amount before estimation.
    """
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=0)

    # Load CPI data and deflate inheritance_amount
    cpi_data = pd.read_csv(path_to_cpi, index_col=0)
    df = deflate_inheritance_amount_for_estimation(df, cpi_data, specs)

    # Set values above 90th percentile to NaN
    p90_threshold = df["inheritance_amount"].quantile(INHERITANCE_QUANTILE_THRESHOLD)
    df.loc[df["inheritance_amount"] > p90_threshold, "inheritance_amount"] = np.nan

    # Prepare data
    df = prepare_inheritance_data(df)

    # Create output directory
    path_to_save_dir = path_to_save_summary.parent
    path_to_save_dir.mkdir(parents=True, exist_ok=True)

    # Define specifications with two care variables
    specifications = [
        # Spec 1-3: No filtering
        {
            "name": "spec1_care_parent_this_year",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": "parent_died_this_year",
            "filter": None,
        },
        {
            "name": "spec2_care_recent_parent_recent",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": "parent_died_recent",
            "filter": None,
        },
        {
            "name": "spec3_care_last_year_parent_last_year",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": "parent_died_last_year",
            "filter": None,
        },
        # Spec 4-6: Filter on parent_died_this_year
        {
            "name": "spec4_care_last_year_filter_parent_this_year",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec7_care_this_year_filter_parent_this_year",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec10_care_recent_filter_parent_this_year",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        # Spec 5,8,11: Filter on parent_died_last_year
        {
            "name": "spec5_care_last_year_filter_parent_last_year",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec8_care_this_year_filter_parent_last_year",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec11_care_recent_filter_parent_last_year",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        # Spec 6,9,12: Filter on parent_died_recent
        {
            "name": "spec6_care_last_year_filter_parent_recent",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec9_care_this_year_filter_parent_recent",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec12_care_recent_filter_parent_recent",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
    ]

    # Run all specifications
    results = {}
    for spec in specifications:
        params = estimate_logit_specification_two_care(
            df=df,
            specs=specs,
            light_care_var=spec["light_var"],
            intensive_care_var=spec["intensive_var"],
            parent_var=spec["parent_var"],
            filter_condition=spec["filter"],
            spec_name=spec["name"],
        )

        # Save results
        output_path = path_to_save_dir / f"{spec['name']}_params.csv"
        params.to_csv(output_path)
        results[spec["name"]] = params

    # Save summary
    print("\n" + "=" * 70)
    print("ALL TWO-CARE PROBABILITY SPECIFICATIONS COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {path_to_save_dir}")

    # Write summary file
    with path_to_save_summary.open("w") as f:
        f.write("Inheritance Probability (Two-Care) Specifications Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total specifications: {len(specifications)}\n")
        f.write(f"Output directory: {path_to_save_dir}\n\n")
        f.write("Specifications:\n")
        for spec in specifications:
            f.write(f"  - {spec['name']}\n")
            f.write(f"    Light care: {spec['light_var']}\n")
            f.write(f"    Intensive care: {spec['intensive_var']}\n")
            f.write(f"    Parent: {spec['parent_var']}\n")
            f.write(f"    Filter: {spec['filter']}\n\n")


@pytask.mark.inheritance
def task_estimate_inheritance_amount_two_care_specifications(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_save_summary: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "inheritance_amount_specs_two_care"
    / "_specs_summary.txt",
) -> None:
    """Estimate 12 specifications with separate light/intensive care variables.

    For ln(inheritance_amount) (OLS).

    CPI deflation is applied to inheritance_amount before estimation.
    """
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=0)

    # Load CPI data and deflate inheritance_amount
    cpi_data = pd.read_csv(path_to_cpi, index_col=0)
    df = deflate_inheritance_amount_for_estimation(df, cpi_data, specs)

    # Set values above 90th percentile to NaN
    p90_threshold = df["inheritance_amount"].quantile(INHERITANCE_QUANTILE_THRESHOLD)
    df.loc[df["inheritance_amount"] > p90_threshold, "inheritance_amount"] = np.nan

    # Prepare data
    df = prepare_inheritance_data(df)

    # Create output directory
    path_to_save_dir = path_to_save_summary.parent
    path_to_save_dir.mkdir(parents=True, exist_ok=True)

    # Define specifications with two care variables
    specifications = [
        # Spec 1-3: No filtering
        {
            "name": "spec1_care_parent_this_year",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": "parent_died_this_year",
            "filter": None,
        },
        {
            "name": "spec2_care_recent_parent_recent",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": "parent_died_recent",
            "filter": None,
        },
        {
            "name": "spec3_care_last_year_parent_last_year",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": "parent_died_last_year",
            "filter": None,
        },
        # Spec 4-6: Filter on parent_died_this_year
        {
            "name": "spec4_care_last_year_filter_parent_this_year",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec7_care_this_year_filter_parent_this_year",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        {
            "name": "spec10_care_recent_filter_parent_this_year",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": None,
            "filter": "parent_died_this_year == 1",
        },
        # Spec 5,8,11: Filter on parent_died_last_year
        {
            "name": "spec5_care_last_year_filter_parent_last_year",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec8_care_this_year_filter_parent_last_year",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        {
            "name": "spec11_care_recent_filter_parent_last_year",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": None,
            "filter": "parent_died_last_year == 1",
        },
        # Spec 6,9,12: Filter on parent_died_recent
        {
            "name": "spec6_care_last_year_filter_parent_recent",
            "light_var": "light_care_last_year",
            "intensive_var": "intensive_care_last_year",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec9_care_this_year_filter_parent_recent",
            "light_var": "light_care",
            "intensive_var": "intensive_care",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
        {
            "name": "spec12_care_recent_filter_parent_recent",
            "light_var": "light_care_recent",
            "intensive_var": "intensive_care_recent",
            "parent_var": None,
            "filter": "parent_died_recent == 1",
        },
    ]

    # Run all specifications
    results = {}
    for spec in specifications:
        params = estimate_ols_specification_two_care(
            df=df,
            specs=specs,
            light_care_var=spec["light_var"],
            intensive_care_var=spec["intensive_var"],
            parent_var=spec["parent_var"],
            filter_condition=spec["filter"],
            spec_name=spec["name"],
        )

        # Save results
        output_path = path_to_save_dir / f"{spec['name']}_params.csv"
        params.to_csv(output_path)
        results[spec["name"]] = params

    # Save summary
    print("\n" + "=" * 70)
    print("ALL TWO-CARE AMOUNT SPECIFICATIONS COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {path_to_save_dir}")

    # Write summary file
    with path_to_save_summary.open("w") as f:
        f.write("Inheritance Amount (Two-Care) Specifications Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total specifications: {len(specifications)}\n")
        f.write(f"Output directory: {path_to_save_dir}\n\n")
        f.write("Specifications:\n")
        for spec in specifications:
            f.write(f"  - {spec['name']}\n")
            f.write(f"    Light care: {spec['light_var']}\n")
            f.write(f"    Intensive care: {spec['intensive_var']}\n")
            f.write(f"    Parent: {spec['parent_var']}\n")
            f.write(f"    Filter: {spec['filter']}\n\n")
