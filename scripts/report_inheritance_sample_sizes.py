from pathlib import Path

import numpy as np
import pandas as pd

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs

INHERITANCE_QUANTILE_THRESHOLD_LOWER = 0.0
INHERITANCE_QUANTILE_THRESHOLD_UPPER = 0.90


def prepare_inheritance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for inheritance regressions.

    This is a local copy of the logic used in
    ``task_estimate_inheritance_soep.prepare_inheritance_data`` but avoids
    importing that module (and thus ``pytask``) so that we can run in
    lightweight analysis scripts.
    """
    # Create squared age term
    df["age_sq"] = df["age"] ** 2

    # Ensure formal care costs dummy exists (fill NaN with 0 if needed)
    if "formal_care_costs_dummy" not in df.columns:
        df["formal_care_costs_dummy"] = 0

    # Enforce mutual exclusivity between informal and formal care (base variables)
    informal_care_mask = (
        (df["light_care"] > 0) | (df["intensive_care"] > 0) | (df["any_care"] > 0)
    )
    df.loc[
        informal_care_mask & (df["formal_care_costs_dummy"] == 1),
        "formal_care_costs_dummy",
    ] = 0

    # Create lagged care variables
    df = df.sort_values(["pid", "syear"])
    df["lagged_light_care"] = df.groupby("pid")["light_care"].shift(1)
    df["lagged_intensive_care"] = df.groupby("pid")["intensive_care"].shift(1)
    df["lagged_any_care"] = df.groupby("pid")["any_care"].shift(1)
    df["lagged_formal_care_costs_dummy"] = df.groupby("pid")[
        "formal_care_costs_dummy"
    ].shift(1)

    # Care indicators for different time periods
    df["any_care_last_year"] = (df["lagged_any_care"] > 0).astype(int)
    df["light_care_last_year"] = (df["lagged_light_care"] > 0).astype(int)
    df["intensive_care_last_year"] = (df["lagged_intensive_care"] > 0).astype(int)
    df["formal_care_costs_dummy_last_year"] = (
        df["lagged_formal_care_costs_dummy"] > 0
    ).astype(int)

    df["light_care_recent"] = (
        (df["light_care"] > 0) | (df["lagged_light_care"] > 0)
    ).astype(int)
    df["intensive_care_recent"] = (
        (df["intensive_care"] > 0) | (df["lagged_intensive_care"] > 0)
    ).astype(int)
    df["any_care_recent"] = ((df["any_care"] > 0) | (df["lagged_any_care"] > 0)).astype(
        int
    )
    df["formal_care_costs_dummy_recent"] = (
        (df["formal_care_costs_dummy"] > 0) | (df["lagged_formal_care_costs_dummy"] > 0)
    ).astype(int)

    # Enforce mutual exclusivity for recent versions
    informal_care_recent_mask = (
        (df["light_care_recent"] > 0)
        | (df["intensive_care_recent"] > 0)
        | (df["any_care_recent"] > 0)
    )
    df.loc[
        informal_care_recent_mask & (df["formal_care_costs_dummy_recent"] == 1),
        "formal_care_costs_dummy_recent",
    ] = 0

    # Parent death indicators
    df["lagged_mother_died"] = df.groupby("pid")["mother_died_this_year"].shift(1)
    df["lagged_father_died"] = df.groupby("pid")["father_died_this_year"].shift(1)

    df["parent_died_this_year"] = (
        (df["mother_died_this_year"] == 1) | (df["father_died_this_year"] == 1)
    ).astype(int)
    df["parent_died_last_year"] = (
        (df["lagged_mother_died"] == 1) | (df["lagged_father_died"] == 1)
    ).astype(int)
    df["parent_died_recent"] = (
        (df["mother_died_this_year"] == 1)
        | (df["lagged_mother_died"] == 1)
        | (df["father_died_this_year"] == 1)
        | (df["lagged_father_died"] == 1)
    ).astype(int)

    # ln(inheritance_amount) for OLS regression (only positive amounts)
    df["ln_inheritance_amount"] = np.nan
    positive_inheritance = df["inheritance_amount"].notna() & (
        df["inheritance_amount"] > 0
    )
    df.loc[positive_inheritance, "ln_inheritance_amount"] = np.log(
        df.loc[positive_inheritance, "inheritance_amount"]
    )

    return df


def main() -> None:
    """Compute sample sizes for inheritance probability and amount regressions."""
    specs = read_and_derive_specs(SRC / "specs.yaml")

    df = pd.read_csv(BLD / "data" / "soep_inheritance_sample.csv", index_col=0)

    # Apply the same 0–90th percentile trimming on inheritance_amount as in
    # the estimation code.
    p10_threshold = df["inheritance_amount"].quantile(
        INHERITANCE_QUANTILE_THRESHOLD_LOWER
    )
    p90_threshold = df["inheritance_amount"].quantile(
        INHERITANCE_QUANTILE_THRESHOLD_UPPER
    )
    df.loc[
        (df["inheritance_amount"] < p10_threshold)
        | (df["inheritance_amount"] > p90_threshold),
        "inheritance_amount",
    ] = np.nan

    # Construct lagged care and parent-death variables and ln(inheritance_amount)
    df = prepare_inheritance_data(df)

    sex_labels = specs["sex_labels"]

    # Logit: probability of positive inheritance (baseline spec used in model)
    print("LOGIT SAMPLE SIZES (probability of positive inheritance):")
    for sex_var, sex_label in enumerate(sex_labels):
        df_sex = df[df["sex"] == sex_var].copy()
        df_sex = df_sex.dropna(
            subset=[
                "inheritance_this_year",
                "age",
                "age_sq",
                "any_care_recent",
                "formal_care_costs_dummy_recent",
                "parent_died_recent",
                "education",
            ]
        )
        print(f"  {sex_label}: N={len(df_sex)}")

    # OLS: ln(inheritance_amount) conditional on parent_died_recent == 1 and positive inheritance
    print("\nOLS SAMPLE SIZES (ln inheritance amount | parent died recent & positive):")
    df_filtered = df[
        (df["parent_died_recent"] == 1) & (df["ln_inheritance_amount"].notna())
    ].copy()
    for sex_var, sex_label in enumerate(sex_labels):
        df_sex = df_filtered[df_filtered["sex"] == sex_var].copy()
        df_sex = df_sex.dropna(
            subset=[
                "ln_inheritance_amount",
                "age",
                "age_sq",
                "light_care_recent",
                "intensive_care_recent",
                "formal_care_costs_dummy_recent",
                "education",
            ]
        )
        print(f"  {sex_label}: N={len(df_sex)}")


if __name__ == "__main__":
    main()
