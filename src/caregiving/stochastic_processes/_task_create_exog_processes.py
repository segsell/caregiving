"""Create exogenous transition probabilities."""

from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
import statsmodels.api as sm
from pytask import Product
from statsmodels.stats.outliers_influence import variance_inflation_factor

from caregiving.config import BLD, SRC
from caregiving.model.shared import (
    BAD_HEALTH,
    FEMALE,
    GOOD_HEALTH,
    MALE,
    RETIREMENT_AGE,
)
from caregiving.utils import save_dict_to_pickle, statsmodels_params_to_dict

MIN_YEAR = 2004
MAX_YEAR = 2017
PARENT_MIN_AGE = 65
MEDIUM_HEALTH = -99


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


@pytask.mark.skip()
def task_create_params_parental_health_transition(
    path_to_save_params_female: Annotated[Path, Product] = BLD
    / "model"
    / "exog_health_transition_female.pkl",
):
    """Health transition probabilities for parents.

    Estimated from SOEP data.


    1 = Excellent
    2 = Very good
    3 = Good
    4 = Fair
    5 = Poor

    1, 2, 3 --> 0
    4, 5 --> 1

    """
    params_female = {
        "mother_bad_health": {
            "bad_health_age": 0.0812661362,
            "bad_health_age_squared": -0.0003235191,
            "bad_health_lagged_good_health": -2.8066457229,
            "bad_health_lagged_bad_health": -0.2541395214,
            "bad_health_constant": -3.0607852445,
        },
    }

    save_dict_to_pickle(params_female, path_to_save_params_female)


@pytask.mark.skip()
def task_create_params_spousal_income(
    path_to_raw_data: Path = BLD / "data" / "estimation_data.csv",
) -> None:
    """Fit linear regression model to predict spousal income."""
    data = pd.read_csv(path_to_raw_data)

    data["age_squared"] = data["age"] ** 2
    data["above_retirement_age"] = np.where(data["age"] >= RETIREMENT_AGE, 1, 0)

    _dat = data[
        [
            "other_income",
            "age",
            "age_squared",
            "above_retirement_age",
            "married",
        ]
    ]
    dat = _dat.dropna()

    regressors = dat[["age", "above_retirement_age", "married"]]
    regressors = sm.add_constant(regressors)

    dat.loc[dat["other_income"] <= 0, "other_income"] = np.finfo(float).eps
    y_log = np.log(dat["other_income"])

    model = sm.OLS(y_log, regressors).fit()

    vif_data = pd.DataFrame()
    vif_data["feature"] = regressors.columns
    vif_data["VIF"] = [
        variance_inflation_factor(regressors.values, i)
        for i in range(regressors.shape[1])
    ]

    return model.params


@pytask.mark.skip()
def task_create_params_exog_other_income(
    path_to_raw_data: Path = BLD / "data" / "estimation_data.csv",
) -> None:
    """Fit linear regression model to predict exogenous other income."""
    data = pd.read_csv(path_to_raw_data)

    data["age_squared"] = data["age"] ** 2
    data["above_retirement_age"] = np.where(data["age"] >= RETIREMENT_AGE, 1, 0)

    _dat = data[
        [
            "other_income",
            "age",
            "age_squared",
            "above_retirement_age",
            "married",
            "high_educ",
        ]
    ]
    dat = _dat.dropna()

    regressors = dat[["age", "above_retirement_age", "married", "high_educ"]]
    regressors = sm.add_constant(regressors)

    dat.loc[dat["other_income"] <= 0, "other_income"] = np.finfo(float).eps
    y_log = np.log(dat["other_income"])

    model = sm.OLS(y_log, regressors).fit()

    vif_data = pd.DataFrame()
    vif_data["feature"] = regressors.columns
    vif_data["VIF"] = [
        variance_inflation_factor(regressors.values, i)
        for i in range(regressors.shape[1])
    ]

    return model.params


@pytask.mark.skip()
def task_create_parental_survival_prob(
    path_to_raw_data: Path = BLD / "data" / "estimation_data.csv",
    path_to_save_female: Annotated[Path, Product] = BLD
    / "model"
    / "exog_survival_prob_female.pkl",
) -> None:
    """Create exogenous survival probabilities from SHARE data.

    The data is used to estimate the probability of a parent being alive in the
    next period

    Args:
        path_to_raw_data (Path): Path to the raw data.
        path_to_save_female (Path): Path to save the survival parameters for females.
        path_to_save_male (Path): Path to save the survival parameters for males.

    Returns:
        None

    """
    dat_all = pd.read_csv(path_to_raw_data)

    dat = _prepare_dependent_variables_health(
        dat_all,
        health="lagged_health",
        age="lagged_age",
    )

    x_mother_with_nans = sm.add_constant(
        dat[
            [
                "mother_lagged_age",
                "mother_lagged_age_squared",
                "mother_lagged_health_bad",
            ]
        ],
    )
    x_mother = x_mother_with_nans.dropna()
    data_mother = dat.dropna(
        subset=[
            "mother_lagged_age",
            "mother_lagged_age_squared",
            "mother_lagged_health_bad",
        ],
    )

    x_female = x_mother[(data_mother["mother_alive"].notna())]
    y_female = data_mother["mother_alive"][(data_mother["mother_alive"].notna())]
    x_female = x_female.reset_index(drop=True)
    y_female = y_female.reset_index(drop=True)

    logit_female = sm.Logit(y_female, x_female).fit()

    params_mother = statsmodels_params_to_dict(
        logit_female.params,
        name_prefix="survival_prob",
        name_constant="mother",
    )

    save_dict_to_pickle(params_mother, path_to_save_female)


@pytask.mark.skip(reason="Reduce to two health states.")
def task_create_parental_survival_prob_good_medium_bad(
    path_to_raw_data: Path = BLD / "data" / "estimation_data.csv",
    path_to_save_female: Annotated[Path, Product] = BLD
    / "model"
    / "exog_survival_prob_good_medium_bad_female.pkl",
    path_to_save_male: Annotated[Path, Product] = BLD
    / "model"
    / "exog_survival_prob_good_medium_bad_male.pkl",
) -> None:
    """Create exogenous survival probabilities from SHARE data.

    The data is used to estimate the probability of a parent being alive in the
    next period

    Args:
        path_to_raw_data (Path): Path to the raw data.
        path_to_save_female (Path): Path to save the survival parameters for females.
        path_to_save_male (Path): Path to save the survival parameters for males.

    Returns:
        None

    """
    dat = pd.read_csv(path_to_raw_data)

    dat = _prepare_dependent_variables_health(
        dat,
        health="lagged_health",
        age="lagged_age",
    )

    x_mother_with_nans = sm.add_constant(
        dat[
            [
                "mother_lagged_age",
                "mother_lagged_age_squared",
                "mother_lagged_health_medium",
                "mother_lagged_health_bad",
            ]
        ],
    )
    x_mother = x_mother_with_nans.dropna()
    data_mother = dat.dropna(
        subset=[
            "mother_lagged_age",
            "mother_lagged_age_squared",
            "mother_lagged_health_medium",
            "mother_lagged_health_bad",
        ],
    )

    x_father_with_nans = sm.add_constant(
        dat[
            [
                "father_lagged_age",
                "father_lagged_age_squared",
                "father_lagged_health_medium",
                "father_lagged_health_bad",
            ]
        ],
    )
    x_father = x_father_with_nans.dropna()
    data_father = dat.dropna(
        subset=[
            "father_lagged_age",
            "father_lagged_age_squared",
            "father_lagged_health_medium",
            "father_lagged_health_bad",
        ],
    )

    x_male = x_father[(data_father["father_alive"].notna())]
    y_male = data_father["father_alive"][(data_father["father_alive"].notna())]
    x_male = x_male.reset_index(drop=True)
    y_male = y_male.reset_index(drop=True)

    x_female = x_mother[(data_mother["mother_alive"].notna())]
    y_female = data_mother["mother_alive"][(data_mother["mother_alive"].notna())]
    x_female = x_female.reset_index(drop=True)
    y_female = y_female.reset_index(drop=True)

    logit_female = sm.Logit(y_female, x_female).fit()
    logit_male = sm.Logit(y_male, x_male).fit()

    params_mother = statsmodels_params_to_dict(
        logit_female.params,
        name_prefix="survival_prob",
        name_constant="mother",
    )
    params_father = statsmodels_params_to_dict(
        logit_male.params,
        name_prefix="survival_prob",
        name_constant="father",
    )

    save_dict_to_pickle(params_mother, path_to_save_female)
    save_dict_to_pickle(params_father, path_to_save_male)


def task_create_params_exog_care_demand_basic(
    path_to_parent_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_parent_couple_data: Path = BLD
    / "data"
    / "share_parent_child_data_couple.csv",
) -> None:
    """Create exogenous care demand probabilities."""
    parent = pd.read_csv(path_to_parent_data)
    couple_raw = pd.read_csv(path_to_parent_couple_data)
    couple = couple_raw[
        (couple_raw["mother_married"] == True) & (couple_raw["father_married"] == True)
    ]

    parent["mother_age"] = parent.loc[parent["gender"] == FEMALE, "age"]
    parent["father_age"] = parent.loc[parent["gender"] == MALE, "age"]

    parent["mother_health"] = parent.loc[parent["gender"] == FEMALE, "health"]
    parent["father_health"] = parent.loc[parent["gender"] == MALE, "health"]

    parent = _prepare_dependent_variables_health(parent)
    couple = _prepare_dependent_variables_health(couple)

    mother = parent[(parent["married"] == False) & (parent["gender"] == FEMALE)].copy()
    father = parent[(parent["married"] == False) & (parent["gender"] == MALE)].copy()

    _cond = [
        (couple["mother_any_care"].isna()) & (couple["father_any_care"].isna()),
        (couple["mother_any_care"] == True) | (couple["father_any_care"] == True),
    ]
    _val = [np.nan, 1]
    couple["any_care"] = np.select(_cond, _val, default=0)

    x_couple_with_nans = sm.add_constant(
        couple[
            [
                "mother_age",
                "mother_age_squared",
                "father_age",
                "father_age_squared",
            ]
        ],
    )
    x_couple = x_couple_with_nans.dropna()
    data_couple = couple.dropna(
        subset=[
            "mother_age",
            "mother_age_squared",
            "father_age",
            "father_age_squared",
        ],
    )

    x_single_mother_with_nans = sm.add_constant(
        mother[
            [
                "mother_age",
                "mother_age_squared",
            ]
        ],
    )
    x_single_mother = x_single_mother_with_nans.dropna()
    data_single_mother = mother.dropna(
        subset=[
            "mother_age",
            "mother_age_squared",
        ],
    )

    x_single_father_with_nans = sm.add_constant(
        father[
            [
                "father_age",
                "father_age_squared",
            ]
        ],
    )
    x_single_father = x_single_father_with_nans.dropna()
    data_single_father = father.dropna(
        subset=[
            "father_age",
            "father_age_squared",
        ],
    )

    # Single father

    x_single_male = x_single_father[
        (data_single_father["any_care"].notna())
        & (data_single_father["gender"] == MALE)
    ]
    y_single_male = data_single_father["any_care"][
        (data_single_father["any_care"].notna())
        & (data_single_father["gender"] == MALE)
    ]
    x_single_male = x_single_male.reset_index(drop=True)
    y_single_male = y_single_male.reset_index(drop=True)

    # Single mother

    x_single_female = x_single_mother[
        (data_single_mother["any_care"].notna())
        & (data_single_mother["gender"] == FEMALE)
    ]
    y_single_female = data_single_mother["any_care"][
        (data_single_mother["any_care"].notna())
        & (data_single_mother["gender"] == FEMALE)
    ]
    x_single_female = x_single_female.reset_index(drop=True)
    y_single_female = y_single_female.reset_index(drop=True)

    # Couple
    x_couple = x_couple[
        (data_couple["any_care"].notna())
        & (data_couple["mother_gender"].notna())
        & (data_couple["father_gender"].notna())
    ]
    y_couple = data_couple["any_care"][
        (data_couple["any_care"].notna())
        & (data_couple["mother_gender"].notna())
        & (data_couple["father_gender"].notna())
    ]
    x_couple = x_couple.reset_index(drop=True)
    y_couple = y_couple.reset_index(drop=True)

    # regress dummy for any care on age and age squared
    # distance to parents?
    # any care in previous period

    logit_single_father = sm.Logit(y_single_male, x_single_male).fit()
    logit_single_mother = sm.Logit(y_single_female, x_single_female).fit()
    logit_couple = sm.Logit(y_couple, x_couple).fit()
    # care demand is zero if no parent is alive

    return logit_single_father.params, logit_single_mother.params, logit_couple.params


@pytask.mark.skip(reason="Reduce to two health states.")
def task_create_params_exog_care_demand(
    path_to_parent_data: Path = BLD / "data" / "share_parent_child_data.csv",
    path_to_parent_couple_data: Path = BLD
    / "data"
    / "share_parent_child_data_couple.csv",
    path_to_save_single_mother: Annotated[Path, Product] = BLD
    / "model"
    / "exog_care_single_mother.pkl",
    path_to_save_single_father: Annotated[Path, Product] = BLD
    / "model"
    / "exog_care_single_father.pkl",
    path_to_save_couple: Annotated[Path, Product] = BLD
    / "model"
    / "exog_care_couple.pkl",
) -> None:
    """Create exogenous care demand probabilities."""
    parent = pd.read_csv(path_to_parent_data)
    couple_raw = pd.read_csv(path_to_parent_couple_data)
    couple = couple_raw[
        (couple_raw["mother_married"] == True) & (couple_raw["father_married"] == True)
    ]

    parent["mother_age"] = parent.loc[parent["gender"] == FEMALE, "age"]
    parent["father_age"] = parent.loc[parent["gender"] == MALE, "age"]

    parent["mother_health"] = parent.loc[parent["gender"] == FEMALE, "health"]
    parent["father_health"] = parent.loc[parent["gender"] == MALE, "health"]

    parent = _prepare_dependent_variables_health(parent)
    couple = _prepare_dependent_variables_health(couple)

    mother = parent[(parent["married"] == False) & (parent["gender"] == FEMALE)].copy()
    father = parent[(parent["married"] == False) & (parent["gender"] == MALE)].copy()

    _cond = [
        (couple["mother_any_care"].isna()) & (couple["father_any_care"].isna()),
        (couple["mother_any_care"] == True) | (couple["father_any_care"] == True),
    ]
    _val = [np.nan, 1]
    couple["any_care"] = np.select(_cond, _val, default=0)

    x_couple_with_nans = sm.add_constant(
        couple[
            [
                "mother_age",
                "mother_age_squared",
                "father_age",
                "father_age_squared",
                "mother_health_medium",
                "mother_health_bad",
                "father_health_medium",
                "father_health_bad",
            ]
        ],
    )
    x_couple = x_couple_with_nans.dropna()
    data_couple = couple.dropna(
        subset=[
            "mother_age",
            "mother_age_squared",
            "father_age",
            "father_age_squared",
            "mother_health_medium",
            "mother_health_bad",
            "father_health_medium",
            "father_health_bad",
        ],
    )

    x_single_mother_with_nans = sm.add_constant(
        mother[
            [
                "mother_age",
                "mother_age_squared",
                "mother_health_medium",
                "mother_health_bad",
            ]
        ],
    )
    x_single_mother = x_single_mother_with_nans.dropna()
    data_single_mother = mother.dropna(
        subset=[
            "mother_age",
            "mother_age_squared",
            "mother_health_medium",
            "mother_health_bad",
        ],
    )

    x_single_father_with_nans = sm.add_constant(
        father[
            [
                "father_age",
                "father_age_squared",
                "father_health_medium",
                "father_health_bad",
            ]
        ],
    )
    x_single_father = x_single_father_with_nans.dropna()
    data_single_father = father.dropna(
        subset=[
            "father_age",
            "father_age_squared",
            "father_health_medium",
            "father_health_bad",
        ],
    )

    # Single father

    x_single_male = x_single_father[
        (data_single_father["any_care"].notna())
        & (data_single_father["gender"] == MALE)
    ]
    y_single_male = data_single_father["any_care"][
        (data_single_father["any_care"].notna())
        & (data_single_father["gender"] == MALE)
    ]
    x_single_male = x_single_male.reset_index(drop=True)
    y_single_male = y_single_male.reset_index(drop=True)

    # Single mother

    x_single_female = x_single_mother[
        (data_single_mother["any_care"].notna())
        & (data_single_mother["gender"] == FEMALE)
    ]
    y_single_female = data_single_mother["any_care"][
        (data_single_mother["any_care"].notna())
        & (data_single_mother["gender"] == FEMALE)
    ]
    x_single_female = x_single_female.reset_index(drop=True)
    y_single_female = y_single_female.reset_index(drop=True)

    # Couple
    x_couple = x_couple[
        (data_couple["any_care"].notna())
        & (data_couple["mother_gender"].notna())
        & (data_couple["father_gender"].notna())
    ]
    y_couple = data_couple["any_care"][
        (data_couple["any_care"].notna())
        & (data_couple["mother_gender"].notna())
        & (data_couple["father_gender"].notna())
    ]
    x_couple = x_couple.reset_index(drop=True)
    y_couple = y_couple.reset_index(drop=True)

    # regress dummy for any care on age and age squared
    # distance to parents?
    # any care in previous period

    logit_single_father = sm.Logit(y_single_male, x_single_male).fit()
    logit_single_mother = sm.Logit(y_single_female, x_single_female).fit()
    logit_couple = sm.Logit(y_couple, x_couple).fit()
    # care demand is zero if no parent is alive

    params_single_mother = statsmodels_params_to_dict(
        logit_single_mother.params,
        name_prefix="exog_care_single",
        name_constant="mother",
    )
    params_single_father = statsmodels_params_to_dict(
        logit_single_father.params,
        name_prefix="exog_care_single",
        name_constant="father",
    )
    params_couple = statsmodels_params_to_dict(
        logit_couple.params,
        name_prefix="exog_care_couple",
    )

    save_dict_to_pickle(params_single_mother, path_to_save_single_mother)
    save_dict_to_pickle(params_single_father, path_to_save_single_father)
    save_dict_to_pickle(params_couple, path_to_save_couple)


def _prepare_dependent_variables_health(data, health="health", age="age"):
    data = data.copy()

    data[f"father_{health}_good"] = np.where(
        data[f"father_{health}"] == GOOD_HEALTH,
        1,
        np.where(data[f"father_{health}"].isna(), np.nan, 0),
    )
    data[f"father_{health}_bad"] = np.where(
        data[f"father_{health}"] == BAD_HEALTH,
        1,
        np.where(data[f"father_{health}"].isna(), np.nan, 0),
    )

    data[f"mother_{health}_good"] = np.where(
        data[f"mother_{health}"] == GOOD_HEALTH,
        1,
        np.where(data[f"mother_{health}"].isna(), np.nan, 0),
    )
    data[f"mother_{health}_bad"] = np.where(
        data[f"mother_{health}"] == BAD_HEALTH,
        1,
        np.where(data[f"mother_{health}"].isna(), np.nan, 0),
    )

    data[f"father_{age}_squared"] = data[f"father_{age}"] ** 2
    data[f"mother_{age}_squared"] = data[f"mother_{age}"] ** 2

    return data


def _task_create_survival_probabilities_stat_office(
    path_to_raw_data: Path = SRC
    / "data"
    / "statistical_office"
    / "12621-0001_Sterbetafel_clean.csv",
) -> None:
    """Create exogenous survival probabilities for parents."""
    data = pd.read_csv(path_to_raw_data)

    # Filter data for the years 2004 to 2017 and age 65 and older
    data_filtered = data[
        (data["year"] >= MIN_YEAR)
        & (data["year"] <= MAX_YEAR)
        & (data["age"] >= PARENT_MIN_AGE)
    ]

    # Prepare the independent variables
    data_filtered = data_filtered.copy()
    data_filtered["age_squared"] = data_filtered["age"] ** 2
    x = sm.add_constant(data_filtered[["age", "age_squared"]])

    # Separate data for males and females within the specified years and age
    x_male = x[data_filtered["male_survival_probability"].notna()]
    y_male = data_filtered["male_survival_probability"][
        data_filtered["male_survival_probability"].notna()
    ]

    x_female = x[data_filtered["female_survival_probability"].notna()]
    y_female = data_filtered["female_survival_probability"][
        data_filtered["female_survival_probability"].notna()
    ]

    logit_male = sm.Logit(y_male, x_male).fit()
    logit_female = sm.Logit(y_female, x_female).fit()

    params_female = statsmodels_params_to_dict(
        logit_female.params,
        name_prefix="survival_prob_female",
    )
    params_male = statsmodels_params_to_dict(
        logit_male.params,
        name_prefix="survival_prob_male",
    )

    return params_female, params_male


def exog_care_demand_probability(
    parental_age,
    parent_alive,
    good_health,
    medium_health,
    bad_health,
    params,
):
    """Create exogenous care demand probabilities.

    Compute based on parent alive. Otherwise zero.
    Done outside?!

    Nested exogenous transitions:
    - First, a parent's health state is determined by their age and lagged health state.

    Args:
        parental_age (int): Age of parent.
        parent_alive (int): Binary indicator of whether parent is alive.
        good_health (int): Binary indicator of good health.
        medium_health (int): Binary indicator of medium health.
        bad_health (int): Binary indicator of bad health.
        params (dict): Dictionary of parameters.

    Returns:
        jnp.ndarray: Array of shape (2,) representing the probabilities of
            no care demand and care demand, respectively.

    """
    survival_prob = predict_survival_probability(parental_age, sex="female")  # mother

    trans_probs_health = exog_health_transition(
        parental_age,
        good_health,
        medium_health,
        bad_health,
        params,
    )
    # parent alive?

    prob_care_good = _exog_care_demand(parental_age, parental_health=0, params=params)
    prob_care_medium = _exog_care_demand(parental_age, parental_health=1, params=params)
    prob_care_bad = _exog_care_demand(parental_age, parental_health=2, params=params)

    _trans_probs_care_demand = jnp.array(
        [prob_care_bad, prob_care_medium, prob_care_good],
    )

    # Non-zero probability of care demand only if parent is alive,
    # weighted by the parent's survival probability
    joint_prob_care_demand = (survival_prob * parent_alive) * (
        trans_probs_health @ _trans_probs_care_demand
    )

    return jnp.array([1 - joint_prob_care_demand, joint_prob_care_demand])


def _exog_care_demand(parental_age, parental_health, params):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    logit = (
        params["exog_care_mother_constant"]
        + params["exog_care_mother_age"] * parental_age
        + params["exog_care_mother_age_squared"] * (parental_age**2)
        + params["exog_care_mother_medium_health"] * (parental_health == MEDIUM_HEALTH)
        + params["exog_care_mother_bad_health"] * (parental_health == BAD_HEALTH)
    )
    return 1 / (1 + np.exp(-logit))


def exog_health_transition(age, good_health, medium_health, bad_health, params):
    """Compute exogenous health transition probabilities.

    Multinomial logit model with three health states: good, medium, bad.

    This function computes the transition probabilities for an individual's health
    state based on their current age, squared age, and lagged health states.
    It uses a set of predefined parameters for medium and bad health states to
    calculate linear combinations, and then applies the softmax function to these
    linear combinations to get the transition probabilities.


    Returns:
        jnp.ndarray: Array of shape (3,) representing the probabilities of
            transitioning to good, medium, and bad health states, respectively.

    """
    age_squared = age**2

    # Linear combination for medium health
    lc_medium_health = (
        params["medium_health"]["medium_health_age"] * age
        + params["medium_health"]["medium_health_age_squared"] * age_squared
        + params["medium_health"]["medium_health_lagged_good_health"] * good_health
        + params["medium_health"]["medium_health_lagged_medium_health"] * medium_health
        + params["medium_health"]["medium_health_lagged_bad_health"] * bad_health
        + params["medium_health"]["medium_health_constant"]
    )

    # Linear combination for bad health
    lc_bad_health = (
        params["bad_health"]["bad_health_age"] * age
        + params["bad_health"]["bad_health_age_squared"] * age_squared
        + params["bad_health"]["bad_health_lagged_good_health"] * good_health
        + params["bad_health"]["bad_health_lagged_medium_health"] * medium_health
        + params["bad_health"]["bad_health_lagged_bad_health"] * bad_health
        + params["bad_health"]["bad_health_constant"]
    )

    linear_comb = np.array([0, lc_medium_health, lc_bad_health])
    transition_probs = softmax(linear_comb)

    return jnp.array([transition_probs[0], transition_probs[1], transition_probs[2]])


def softmax(lc):
    """Compute the softmax of each element in an array of linear combinations.

    The softmax function is applied to an array of linear combination values (lc)
    to calculate the probabilities of each class in a multinomial logistic
    regression model.
    This function is typically used for multi-class classification problems.

    Args:
        lc (np.ndarray): An array of linear combination values. This can be a 1D array
            representing linear combinations for each class in a single data point,
            or a 2D array representing multiple data points.

    Returns:
        np.ndarray: An array of the same shape as `lc` where each value is transformed
            into the probability of the corresponding class, ensuring that the sum of
            probabilities across classes (for each data point if 2D) equals 1.

    Example:
    >>> lc = np.array([0, 1, 2])
    >>> softmax(lc)
    array([0.09003057, 0.24472847, 0.66524096])

    Note:
    - The function applies np.exp to each element in `lc` and then normalizes so that
      the sum of these exponentials is 1.
    - For numerical stability, the maximum value in each set of linear combinations
      is subtracted from each linear combination before exponentiation.

    """
    e_lc = np.exp(lc - np.max(lc))  # Subtract max for numerical stability
    return e_lc / e_lc.sum(axis=0)


def predict_other_income(age, married, high_educ, params):
    """Predict other income based on log-lin regression."""
    log_other_income = (
        params["other_income_const"]
        + params["other_income_age"] * age
        + params["other_income_above_retirement_age"] * (age >= RETIREMENT_AGE)
        + params["other_income_married"] * married
        + params["other_income_high_educ"] * high_educ
    )

    return np.exp(log_other_income)


def predict_care_demand(
    age,
    sex,
    lagged_care,
    coefs_single_father,
    coefs_single_mother,
    coefs_couple,
):
    """Predicts the care demand based on logit parameters.

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    if sex.lower() == "male":
        coefs = coefs_single_father
    elif sex.lower() == "female":
        coefs = coefs_single_mother
    else:
        coefs = coefs_couple

    # Logit prediction
    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2) + coefs[3] * lagged_care
    return 1 / (1 + np.exp(-logit))


def predict_survival_probability(age, sex):
    """Predicts the survival probability based on logit parameters.

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    coefs_male = np.array(
        [11.561515476144223, -0.11058331994203506, -1.0998977981246952e-05],
    )
    coefs_female = np.array(
        [17.01934835131644, -0.21245937682111807, 0.00047537366767865137],
    )

    if sex.lower() == "male":
        coefs = coefs_male
    elif sex.lower() == "female":
        coefs = coefs_female

    # Logit prediction
    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2)
    return 1 / (1 + np.exp(-logit))
