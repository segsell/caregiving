"""Initial conditions."""

from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytask import Product

from caregiving.config import BLD
from caregiving.moments._task_create_empirical_moments import deflate_income_and_wealth

BASE_YEAR = 2015

MALE = 1
FEMALE = 2

START_AGE = 40
MIN_AGE = 50
MAX_AGE = 65

AGE_50 = 50
AGE_53 = 53
AGE_56 = 56
AGE_59 = 59
AGE_62 = 62

AGE_55 = 55
AGE_60 = 60
AGE_65 = 65

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_initial_conditions(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    path_wealth: Annotated[Path, Product] = BLD
    / "moments"
    / "initial_wealth_at_age_50.csv",
    path_discrete: Annotated[Path, Product] = BLD
    / "moments"
    / "_initial_discrete_conditions_SHARE_at_age_50.csv",
) -> None:
    """Create initial conditions at age 50.

    State variables:
    - age
    - married
    - has_sister or has_sibling
    - health mother = good
    - health father = good
    - mother alive
    - father alive

    Choices
    - share not working part time, full time
    - share informal care
    - share formal care


    Usage:

    _mother_age = draw_parental_age(
        seed=2024,
        n_agents=100,
        mean=initial_mother_age_mean,
        std_dev=initial_mother_age_std,
    )
    _mother_age_clipped = np.clip(_mother_age - 10, START_AGE + 18, 100)

    """
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    dat = deflate_income_and_wealth(dat, cpi_data)

    weight = "hh_weight"
    intensive_care_var = "intensive_care_no_other"

    dat["has_sister_weighted"] = dat["has_sister"] * dat[weight]
    dat["has_sibling_weighted"] = dat["has_sibling"] * dat[weight]
    dat["married_unweighted"] = dat["married"] / dat[weight]
    dat["informal_care_weighted"] = dat[intensive_care_var] * dat[weight]
    dat["mother_health_unweighted"] = dat["mother_health"] / dat[weight]
    dat["father_health_unweighted"] = dat["father_health"] / dat[weight]
    dat["mother_alive_unweighted"] = dat["mother_alive"] / dat[weight]
    dat["father_alive_unweighted"] = dat["father_alive"] / dat[weight]

    dat["father_health_good"] = dat["father_health_unweighted"] == GOOD_HEALTH
    dat["father_health_medium"] = dat["father_health_unweighted"] == MEDIUM_HEALTH
    dat["father_health_bad"] = dat["father_health_unweighted"] == BAD_HEALTH
    dat["father_health_good_weighted"] = dat["father_health_good"] * dat[weight]
    dat["father_health_medium_weighted"] = dat["father_health_medium"] * dat[weight]
    dat["father_health_bad_weighted"] = dat["father_health_bad"] * dat[weight]

    dat["mother_health_good"] = dat["mother_health_unweighted"] == GOOD_HEALTH
    dat["mother_health_medium"] = dat["mother_health_unweighted"] == MEDIUM_HEALTH
    dat["mother_health_bad"] = dat["mother_health_unweighted"] == BAD_HEALTH
    dat["mother_health_good_weighted"] = dat["mother_health_good"] * dat[weight]
    dat["mother_health_medium_weighted"] = dat["mother_health_medium"] * dat[weight]
    dat["mother_health_bad_weighted"] = dat["mother_health_bad"] * dat[weight]

    employment_shares_age_50 = get_employment_shares_age_50_soep()

    share_has_sister = _get_share_at_min_age(dat, weight, moment="has_sister_weighted")
    share_has_sibling = _get_share_at_min_age(
        dat,
        weight,
        moment="has_sibling_weighted",
    )
    share_married = _get_share_at_min_age(dat, weight, moment="married")
    share_informal_care = _get_share_at_min_age(
        dat,
        weight,
        moment="informal_care_weighted",
    )

    share_mother_alive = _get_share_at_min_age(dat, weight, moment="mother_alive")
    share_father_alive = _get_share_at_min_age(dat, weight, moment="father_alive")

    (
        mother_good_health,
        mother_medium_health,
        mother_bad_health,
        father_good_health,
        father_medium_health,
        father_bad_health,
    ) = get_initial_parental_health(
        dat,
        weight=weight,
    )

    (
        initial_mother_age_mean,
        initial_mother_age_std,
        initial_father_age_mean,
        initial_father_age_std,
    ) = get_inital_parent_age_mean_and_std_dev(
        dat,
        age=MIN_AGE,
        weight=weight,
    )

    initial_discrete_conditions = pd.Series(
        {
            "share_has_sister": share_has_sister,
            "share_has_sibling": share_has_sibling,
            "share_married": share_married,
            "share_not_working": employment_shares_age_50["not_working"],
            "share_part_time": employment_shares_age_50["part_time"],
            "share_full_time": employment_shares_age_50["full_time"],
            "share_informal_care": share_informal_care,
            "mother_age_mean": initial_mother_age_mean,
            "mother_age_std": initial_mother_age_std,
            "father_age_mean": initial_father_age_mean,
            "father_age_std": initial_father_age_std,
            "share_mother_alive": share_mother_alive,
            "share_father_alive": share_father_alive,
            "mother_good_health": mother_good_health,
            "mother_medium_health": mother_medium_health,
            "mother_bad_health": mother_bad_health,
            "father_good_health": father_good_health,
            "father_medium_health": father_medium_health,
            "father_bad_health": father_bad_health,
        },
    )

    dat["hnetw_unweighted"] = dat["hnetw"] / dat[weight]
    initial_wealth = dat.loc[dat["age"] == MIN_AGE, "hnetw_unweighted"].dropna()

    initial_wealth.to_csv(path_wealth, index=False)

    initial_discrete_conditions.name = "moment"
    initial_discrete_conditions.to_csv(path_discrete, index=True)


# ==============================================================================


def draw_random_array(seed, n_agents, values, probabilities):
    """Draw a random array with given probabilities.

    Usage:

    seed = 2024
    n_agents = 10_000

    # Parameters
    values = jnp.array([-1, 0, 1, 2])  # Values to choose from
    probabilities = jnp.array([0.3, 0.3, 0.2, 0.2])  # Corresponding probabilities

    table(pd.DataFrame(random_array)[0]) / 1000

    """
    key = jax.random.PRNGKey(seed)  # Initialize a random key
    return jax.random.choice(key, values, shape=(n_agents,), p=probabilities)


def draw_parental_age(seed, n_agents, mean, std_dev):
    """Draw discrete parental age."""
    key = jax.random.PRNGKey(seed)

    # Sampling standard normal values
    sample_standard_normal = jax.random.normal(key, (n_agents,))

    # Scaling and shifting to get the desired mean and standard deviation, then rounding
    return jnp.round(mean + std_dev * sample_standard_normal).astype(jnp.int32)


def draw_random_sequence_from_array(seed, arr, n_agents):
    """Draw a random sequence from an array.

    rand = draw_random_sequence_from_array(     seed=2024,     n_agents=10_000,
    arr=jnp.array(initial_wealth), )

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, arr, shape=(n_agents,), replace=True)


# ==============================================================================


def get_inital_parent_age_mean_and_std_dev(dat, age, weight):
    """Get mean and standard deviation of parental age.

    initial_mother_age = draw_parental_age( seed=2024, n_agents=10_000,
    mean=initial_mother_age_mean,

    std_dev=initial_mother_age_std, )

    """
    # parental age

    dat["mother_age_unweighted"] = dat["mother_age"] / dat[weight]
    dat["father_age_unweighted"] = dat["father_age"] / dat[weight]

    dat["mother_age_unweighted"] = np.where(
        dat["mother_age_unweighted"] < MIN_AGE + 16,
        np.nan,
        dat["mother_age_unweighted"],
    )
    dat["father_age_unweighted"] = np.where(
        dat["father_age_unweighted"] < MIN_AGE + 16,
        np.nan,
        dat["father_age_unweighted"],
    )

    mother_age_mean, mother_age_std = get_mean_and_std_parental_age(
        dat,
        moment="mother_age_unweighted",
        age=age,
    )
    father_age_mean, father_age_std = get_mean_and_std_parental_age(
        dat,
        moment="father_age_unweighted",
        age=age,
    )

    return mother_age_mean, mother_age_std, father_age_mean, father_age_std


def get_mean_and_std_parental_age(dat, moment, age):
    mean = dat.loc[(dat["age"] == age), moment].mean()
    var = dat.loc[(dat["age"] == age), moment].std()

    return mean, var


def _get_share_at_min_age(dat, weight, moment):
    return (
        dat.loc[(dat["age"] == MIN_AGE), moment].sum()
        / dat.loc[(dat["age"] == MIN_AGE), weight].sum()
    )


def _get_share_parental_health_at_min_age(dat, weight, moment):
    return (
        dat.loc[(dat["age"] == MIN_AGE), moment].sum()
        / dat.loc[
            (dat["age"] == MIN_AGE) & (dat[moment].notna()),
            weight,
        ].sum()
    )


def _get_share_parental_health_at_min_age_unweighted(dat, moment, outcome):
    return (
        dat[(dat["age"] == MIN_AGE) & (dat[moment] == outcome)].shape[0]
        / dat[(dat["age"] == MIN_AGE) & (dat[moment].notna())].shape[0]
    )


def get_share_at_min_age(dat, weight, moment):
    return (
        dat[(dat["age"] == MIN_AGE) & (dat[moment] == True)].shape[0]
        / dat.loc[(dat["age"] == MIN_AGE), weight].shape[0]
    )


def get_share_at_age_interval(dat, weight, moment, age_bottom, age_top):
    return (
        dat.loc[
            (dat["age"] >= age_bottom)
            & (dat["age"] <= age_top)
            & (dat[moment] == True),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] >= age_bottom) & (dat["age"] <= age_top),
            weight,
        ].sum()
    )


def get_employment_shares_age_50_soep():
    """Get employment shares at age 50 from SOEP.

    (Pdb++)  0.2782375 + 0.35092820 + 0.37083427
    0.9999999700000001

    """
    return pd.Series(
        {
            "not_working": 0.2782375,
            "part_time": 0.35092820,
            "full_time": 0.37083427,
        },
    )


def _create_distribution_plot_parental_age(dat, parent, color="purple"):
    """Create distribution plot for parental age.

    # Plot the distribution of "mother_age_unweighted" using a histogram
    plt.hist(dat[f'{parent}_age_unweighted'].dropna(), bins=20, edgecolor='k',
    color='purple') plt.xlabel(f"{parent.capitalize()} Age") plt.ylabel('Frequency')
    plt.title(f"{parent.capitalize()} Age") plt.show()

    """
    # Plot the distribution of "mother_age_unweighted" using Seaborn
    sns.histplot(
        dat[f"{parent}_age_unweighted"].dropna(),
        bins=40,
        kde=True,
        color=color,
    )
    plt.xlabel(f"{parent.capitalize()} Age")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {parent.capitalize()} Age")
    plt.show()


def _plot_dist(sample):
    sns.histplot(sample, bins=40, kde=True, color="purple")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Distribution of Age")
    plt.show()


def _plot_density_distribution(
    sample,
    xlabel="Value",
    title="Density Distribution",
):
    """Plot the density distribution of a specified column in a pandas DataFrame.

    Args:
        sample (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Value'.
        title (str, optional): The title of the plot. Defaults to
            'Density Distribution'.

    """
    sns.kdeplot(sample, shade=True)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.show()


def get_initial_parental_health(dat, weight):
    share_mother_good_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="mother_health_good_weighted",
    )
    share_mother_medium_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="mother_health_medium_weighted",
    )
    share_mother_bad_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="mother_health_bad_weighted",
    )

    share_father_good_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="father_health_good_weighted",
    )
    share_father_medium_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="father_health_medium_weighted",
    )
    share_father_bad_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="father_health_bad_weighted",
    )

    sum_mother_health = (
        share_mother_good_health + share_mother_medium_health + share_mother_bad_health
    )
    sum_father_health = (
        share_father_good_health + share_father_medium_health + share_father_bad_health
    )

    mother_good_health = share_mother_good_health / sum_mother_health
    mother_medium_health = share_mother_medium_health / sum_mother_health
    mother_bad_health = share_mother_bad_health / sum_mother_health

    father_good_health = share_father_good_health / sum_father_health
    father_medium_health = share_father_medium_health / sum_father_health
    father_bad_health = share_father_bad_health / sum_father_health

    assert np.allclose(
        mother_good_health + mother_medium_health + mother_bad_health,
        # share_mother_alive,
        1,
    )
    assert np.allclose(
        father_good_health + father_medium_health + father_bad_health,
        # share_father_alive,
        1,
    )

    return (
        mother_good_health,
        mother_medium_health,
        mother_bad_health,
        father_good_health,
        father_medium_health,
        father_bad_health,
    )
