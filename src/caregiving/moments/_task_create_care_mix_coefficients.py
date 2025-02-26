"""Care mix regressions on parent data."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product
from sklearn.linear_model import LogisticRegression

from caregiving.config import BLD
from caregiving.model.shared import AGE_65, BAD_HEALTH, FEMALE


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


@pytask.mark.skip()
def task_create_care_mix_moments(
    path_to_hh_weight: Path = BLD  # noqa: ARG001
    / "data"
    / "estimation_data_hh_weight.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_save: Annotated[Path, Product] = BLD  # noqa: ARG001
    / "moments"
    / "care_mix_regressions.csv",
) -> None:
    """Create care mix moments for parent data.

    (Pdb++) coeff_no_care
        Feature  Coefficient
    0  Intercept     1.105644
    1      70-74    -0.493338
    2      75-79    -1.267638
    3      80-84    -0.945208
    4        85+    -1.735684
    (Pdb++) coeff_combination_care
        Feature  Coefficient
    0  Intercept    -1.328901
    1      70-74     0.570636
    2      75-79     1.077418
    3      80-84     1.758641
    4        85+     2.411376
    (Pdb++) coeff_pure_informal_care
        Feature  Coefficient
    0  Intercept    -0.045870
    1      70-74    -0.029986
    2      75-79     0.121178
    3      80-84     0.478167
    4        85+    -0.172910

    """
    parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    weight = "hh_weight"
    # intensive_care_var = "intensive_care_no_other"

    parent = parent_hh_weight.copy()

    parent["informal_care_child_weighted"] = (
        parent["informal_care_child"] * parent[weight]
    )
    parent["home_care_weighted"] = parent["home_care"] * parent[weight]
    parent["formal_care_weighted"] = parent["formal_care"] * parent[weight]
    parent["combination_care_weighted"] = parent["combination_care"] * parent[weight]
    parent["no_combination_care_weighted"] = (
        parent["no_combination_care"] * parent[weight]
    )

    parent["only_informal_weighted"] = parent["only_informal"] * parent[weight]
    parent["only_home_care_weighted"] = parent["only_home_care"] * parent[weight]

    # dat["no_intensive_informal_weighted"] = dat["no_intensive_informal"] * dat[weight]
    # dat["intensive_care_no_other_weighted"] = (
    #     dat["intensive_care_no_other"] * dat[weight]
    # )
    # intensive_care_var_weighted = "intensive_care_no_other_weighted"

    parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    parent["no_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )
    parent["only_formal_weighted"] = parent["only_formal"] * parent[weight]
    parent["no_only_formal_weighted"] = parent["no_only_formal"] * parent[weight]
    parent["no_only_informal_weighted"] = parent["no_only_informal"] * parent[weight]

    mother = parent[(parent["gender"] == FEMALE)]

    parent["only_informal_care_child_weighted"] = (
        parent["informal_care_child"] * parent[weight]
    )
    parent["no_only_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )
    parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    parent["no_formal_care_weighted"] = parent["no_formal_care"] * parent[weight]
    parent["no_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )

    # Parent-child sample
    mother = mother[mother["health"] == BAD_HEALTH]
    mother = mother[mother["age"] >= AGE_65]
    mother = mother[mother["has_daughter"] == 1]

    age_labels = ["65-69", "70-74", "75-79", "80-84", "85+"]

    mother["age_bin"] = pd.cut(
        mother["age"],
        bins=[65, 70, 75, 80, 85, np.inf],
        right=False,
        labels=age_labels,
        include_lowest=True,
    )
    age_dummies = pd.get_dummies(mother["age_bin"], drop_first=True)
    mother = pd.concat([mother, age_dummies], axis=1)

    coeff_no_care = weighted_logistic_regression(  # noqa: F841
        mother,
        age_dummies,
        outcome="no_care",
        weight=weight,
    )
    coeff_pure_informal_care = weighted_logistic_regression(  # noqa: F841
        mother,
        age_dummies,
        outcome="informal_care_child_no_comb",
        weight=weight,
    )
    coeff_combination_care = weighted_logistic_regression(  # noqa: F841
        mother,
        age_dummies,
        outcome="combination_care",
        weight=weight,
    )

    # features = age_dummies.columns.tolist()

    # X = mother[features]
    # y = mother["combination_care_weighted"]
    # # data = pd.concat([X, y], axis=1)
    # data = pd.concat([X, y, mother[weight]], axis=1)
    # data = data.dropna()

    # X = data[features]
    # y = data["combination_care_weighted"]
    # weights = data[weight]

    # # Initialize the Logistic Regression model with weights
    # # Train the model with weights

    # X = X.mul(weights, axis=0)

    # model = LogisticRegression()
    # model.fit(X, y)

    # # Make predictions (if needed)
    # y_pred = model.predict(X)

    # manual = pd.DataFrame(
    #     {
    #         "Feature": ["Intercept"] + features,
    #         "Coefficient": [model.intercept_[0]] + list(model.coef_[0]),
    #     }
    # )
    # breakpoint()


def weighted_logistic_regression(mother, age_dummies, outcome, weight):
    # Filter the dataset where "health" == 'BAD_HEALTH'
    # mother_filtered = mother[mother["health"] == "BAD_HEALTH"]

    # # Keep people aged 65 and older
    # mother_filtered = mother_filtered[mother_filtered["age"] >= 65]

    # # Create 5-year age bins from [65, 70), [70, 75) etc. up to [80, 85).
    # # The final age bin is 80+
    # age_bins = [65, 70, 75, 80, np.inf]
    # age_labels = ["65-69", "70-74", "75-79", "80+"]
    # mother_filtered["age_bin"] = pd.cut(
    #     mother_filtered["age"],
    #     bins=age_bins,
    #     right=False,
    #     labels=age_labels,
    #     include_lowest=True,
    # )

    # # Create dummy variables for age bins
    # age_dummies = pd.get_dummies(mother_filtered["age_bin"], drop_first=True)

    # # Concatenate the dummies with the original DataFrame
    # mother_filtered = pd.concat([mother_filtered, age_dummies], axis=1)

    # Define the features (age dummies) and the target (y_variable)
    features = age_dummies.columns.tolist()
    X = mother[features]  # noqa: N806
    y = mother[outcome]

    # Combine X, y, and weights into a single DataFrame to drop rows with NaNs
    data = pd.concat([X, y, mother[weight]], axis=1)

    # Drop rows with any NaN values
    data = data.dropna()

    # Separate the cleaned data back into X, y, and weights
    X = data[features]  # noqa: N806
    y = data[outcome]
    weights = data[weight]

    # Initialize the Logistic Regression model with weights
    model = LogisticRegression(class_weight="balanced")

    # Train the model with weights
    model.fit(X, y, sample_weight=weights)

    return pd.DataFrame(
        {
            "Feature": ["Intercept", *features],
            "Coefficient": [model.intercept_[0], *model.coef_[0]],
        },
    )
