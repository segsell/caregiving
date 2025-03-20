"""Estimate probabilities of limitations with Activities of Daily Living.

On SHARE parent-child sample.

"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import FEMALE, MALE


def task_estimate_limitations_with_adl_categories(
    path_to_load_specs: Path = SRC / "specs.yaml",
    # path_to_start_params: Path = SRC
    # / "start_params_and_bounds"
    # / "start_params.yaml",
    path_to_parent_child_sample: Path = BLD / "data" / "parent_child_data.csv",
    path_to_save_adl_probabilities: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_params.csv",
):

    df = pd.read_csv(path_to_parent_child_sample)

    model_men, model_women = run_multinomial_by_gender(df)
    results_dict = get_nested_params(model_men, model_women)

    df_men = pivot_model_params(model_men, "Men")
    df_women = pivot_model_params(model_women, "Women")
    df_combined = pd.concat([df_men, df_women], ignore_index=True)

    df_combined.to_csv(path_to_save_adl_probabilities, index=False)

    # Check identity
    test_params_equality(results_dict, df_combined, tol=1e-10)


def run_multinomial_by_gender(df):
    """
    Run separate multinomial logit regressions of 'adl_cat' on
    age, age^2, and health, by gender. We assume:
       - gender == 1 => men
       - gender == 2 => women
    The 'health' variable is treated as categorical (with levels 0,1,2).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        ['adl_cat', 'age', 'health', 'gender'].

    Returns
    -------
    (model_men, model_women)
        Where each is a fitted multinomial logit model.
    """
    # Subset data for men and women
    dat_men = df[df["gender"] == MALE].copy()
    dat_women = df[df["gender"] == FEMALE].copy()

    # Define the formula
    formula = "adl_cat ~ age + I(age**2) + C(health)"

    # Fit the model for men
    model_men = smf.mnlogit(formula, data=dat_men).fit()
    print("Results for men (gender == 1):")
    print(model_men.summary())

    # Fit the model for women
    model_women = smf.mnlogit(formula, data=dat_women).fit()
    print("\nResults for women (gender == 2):")
    print(model_women.summary())

    return model_men, model_women


def get_nested_params(model_men, model_women):
    """Convert the model.params into a nested dictionary.

    {
      "men": {
        "category_1": {
           "intercept": <coef>,
           "medium_health": <coef>,
           "bad_health": <coef>,
           "age": <coef>,
           "age_squared": <coef>
        },
        "category_2": {...},
        "category_3": {...}
      },
      "women": {
        "category_1": {...},
        "category_2": {...},
        "category_3": {...}
      }
    }

    Notes:
      - We map numeric categories (0,1,2,...) to "category_{X+1}".
      - We rename row-index keys:
         "Intercept"          -> "intercept"
         "C(health)[T.1.0]"   -> "medium_health"
         "C(health)[T.2.0]"   -> "bad_health"
         "age"                -> "age"
         "I(age ** 2)"        -> "age_squared"
    """

    # Helper dictionary to rename row-index keys
    rename_map = {
        "Intercept": "const",
        "age": "age",
        "I(age ** 2)": "age_sq",
        "C(health)[T.1.0]": "medium_health",
        "C(health)[T.2.0]": "bad_health",
    }

    def process_model_params(model_params: pd.DataFrame):
        """Convert model_params (a DataFrame) to a nested dict.

        {
          "category_1": {var_name: coefficient, ...},
          "category_2": {...},
          "category_3": {...},
          ...
        }
        """
        outer_dict = {}

        # model_params.columns are the categories (often 0,1,2,...)
        for cat_col in model_params.columns:
            # Create the new category label
            cat_name = f"category_{int(cat_col) + 1}"

            cat_dict = {}
            for var_name in model_params.index:
                # Rename the variable
                new_var_name = rename_map.get(var_name, var_name)
                # Extract coefficient
                coef_value = model_params.loc[var_name, cat_col]
                cat_dict[new_var_name] = coef_value

            outer_dict[cat_name] = cat_dict

        return outer_dict

    men_nested = process_model_params(model_men.params)
    women_nested = process_model_params(model_women.params)

    return {"men": men_nested, "women": women_nested}


def pivot_model_params(model, sex_label):
    """
    Given a fitted mnlogit model (model.params) and a label for sex ("Men"/"Women"),
    return a WIDE DataFrame with columns:
       sex, adl_cat, const, age, age_sq, medium_health, bad_health
    where each row corresponds to one ADL category.
    """

    # 1) Copy the param matrix: shape [param_names x categories], e.g.:
    #    columns = [0,1,2], index = ["Intercept","age","C(health)[T.1.0]", ...]
    df = model.params.copy()

    # 2) Transpose so that categories become the row index
    #    Now shape: [categories x param_names].
    df = df.T

    # 3) Rename the columns so they match the final desired column names
    rename_dict = {
        "Intercept": "const",
        "age": "age",
        "I(age ** 2)": "age_sq",
        "C(health)[T.1.0]": "medium_health",
        "C(health)[T.2.0]": "bad_health",
    }
    df.rename(columns=rename_dict, inplace=True)

    # 4) The row index is the numeric ADL category from the model (e.g. 0,1,2,...).
    #    We want to rename 0->1, 1->2, 2->3, etc.
    df.index = df.index + 1
    df.index.name = "adl_cat"

    # 5) Turn the row index into a regular column
    df.reset_index(inplace=True)  # Now "adl_cat" is a column

    # 6) Add the sex column at the front
    df.insert(0, "sex", sex_label)

    # 7) Make sure we have the final columns in the EXACT order desired
    desired_cols = [
        "sex",
        "adl_cat",
        "const",
        "age",
        "age_sq",
        "medium_health",
        "bad_health",
    ]

    # Ensure any missing columns are added (as NaN) in case the model didn't have them
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only the desired columns, in the exact order
    df = df[desired_cols]

    return df


def test_params_equality(results_dict, df, tol=1e-10):
    """
    Check that each row of the wide DataFrame (with columns
    ["sex","adl_cat","const","age","age_sq","medium_health","bad_health"])
    matches the corresponding dictionary entry in results_dict.

    Parameters
    ----------
    results_dict : dict
        Nested dict of the form:
        {
          "men": {
            "category_1": {"const":float, "age":float, "age_sq":float,
                "medium_health":float, "bad_health":float},
            "category_2": {...},
            ...
          },
          "women": {
            "category_1": {...},
            ...
          }
        }

    df : pandas.DataFrame
        The wide DataFrame read from CSV with columns:
        ["sex","adl_cat","const","age","age_sq","medium_health","bad_health"].

    tol : float
        Numerical tolerance for comparing floats (uses np.isclose with atol=tol).

    Raises
    ------
    AssertionError
        If any entry does not match between the DataFrame and the dictionary
        within the numerical tolerance.

    Returns
    -------
    None
    """
    required_cols = [
        "sex",
        "adl_cat",
        "const",
        "age",
        "age_sq",
        "medium_health",
        "bad_health",
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Go row by row in the DataFrame
    for i, row in df.iterrows():
        # Convert "Men" -> "men", "Women" -> "women" so we can index results_dict
        sex_key = row["sex"].lower()  # 'men' or 'women'
        cat_key = f"category_{int(row['adl_cat'])}"

        # Retrieve the dict entry for that sex and category
        dict_values = results_dict[sex_key][cat_key]

        # For each coefficient name in the dictionary, compare numeric values
        for var in ("const", "age", "age_sq", "medium_health", "bad_health"):
            val_df = row[var]
            val_dict = dict_values[var]

            # Use np.isclose for float comparisons
            if not np.isclose(val_df, val_dict, atol=tol, rtol=0):
                msg = (
                    f"Mismatch at DataFrame row {i}: "
                    f"sex={row['sex']}, adl_cat={row['adl_cat']}, variable={var}\n"
                    f"  DataFrame value = {val_df}\n"
                    f"  Dictionary value = {val_dict}\n"
                )
                raise AssertionError(msg)

    print("All checks passed!")
