import jax.numpy as jnp

from caregiving.model.shared import (
    BAD_HEALTH,
    DEAD,
    GOOD_HEALTH,
    MEDIUM_HEALTH,
    is_bad_health,
    is_full_time,
    is_medium_health,
    is_part_time,
    is_unemployed,
)

# ==============================================================================
# Exogenous processes
# ==============================================================================


def prob_part_time_offer(period, lagged_choice, options, params):
    """Compute logit probability of part time offer."""
    age = options["start_age"] + period

    logit = (
        params["part_time_constant"]
        + params["part_time_not_working_last_period"] * is_unemployed(lagged_choice)
        # + params["part_time_high_education"] * high_educ # noqa: ERA001
        + params["part_time_above_retirement_age"] * (age >= options["retirement_age"])
    )
    prob_logit = 1 / (1 + jnp.exp(-logit))

    prob_part_time = (
        is_part_time(lagged_choice) * 1 + (1 - is_part_time(lagged_choice)) * prob_logit
    )
    prob_part_time = (age < options["max_ret_age"]) * prob_part_time

    return jnp.array([1 - prob_part_time, prob_part_time])


def prob_full_time_offer(period, lagged_choice, options, params):
    """Compute logit probability of full time offer.

    _prob = jnp.exp(logit) / (1 + jnp.exp(logit))

    """
    age = options["start_age"] + period

    logit = (
        params["full_time_constant"]
        + params["full_time_not_working_last_period"] * is_unemployed(lagged_choice)
        # + params["full_time_high_education"] * high_educ # noqa: ERA001
        + params["full_time_above_retirement_age"] * (age >= options["retirement_age"])
    )
    prob_logit = 1 / (1 + jnp.exp(-logit))

    prob_full_time = (
        is_full_time(lagged_choice) * 1 + (1 - is_full_time(lagged_choice)) * prob_logit
    )
    prob_full_time = (age < options["max_ret_age"]) * prob_full_time

    return jnp.array([1 - prob_full_time, prob_full_time])


# =====================================================================================
# Parental survival and health
# =====================================================================================


def prob_survival_mother_medium_bad(period, mother_health, mother_alive, options):
    """Predicts the survival probability based on logit parameters.

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

    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2)

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    mother_age = options["mother_start_age"] + period

    logit = (
        options["survival_prob_mother_constant"]
        + options["survival_prob_mother_lagged_age"] * mother_age
        + options["survival_prob_mother_lagged_age_squared"] * (mother_age**2)
        + options["survival_prob_mother_lagged_health_medium"]
        * is_medium_health(mother_health)
        + options["survival_prob_mother_lagged_health_bad"]
        * is_bad_health(mother_health)
    )
    prob_logit = 1 / (1 + jnp.exp(-logit))

    prob_survival = mother_alive * prob_logit

    return jnp.array([1 - prob_survival, prob_survival])


def prob_survival_mother(period, mother_health, mother_alive, options):
    """Predicts the survival probability based on logit parameters.

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

    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2)

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    mother_age = options["mother_start_age"] + period

    logit = (
        options["survival_prob_mother_constant"]
        + options["survival_prob_mother_lagged_age"] * mother_age
        + options["survival_prob_mother_lagged_age_squared"] * (mother_age**2)
        + options["survival_prob_mother_lagged_health_bad"]
        * is_bad_health(mother_health)
    )
    prob_logit = 1 / (1 + jnp.exp(-logit))

    prob_survival = mother_alive * prob_logit

    return jnp.array([1 - prob_survival, prob_survival])


def exog_health_transition_mother_with_survival(period, mother_health, options):
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
    mother_age = options["mother_start_age"] + period
    mother_age_squared = mother_age**2

    good_health = mother_health == GOOD_HEALTH
    bad_health = mother_health == BAD_HEALTH
    alive = mother_health != DEAD

    mother_survival_prob = prob_survival_mother(
        period=period,
        mother_health=mother_health,
        mother_alive=alive,
        options=options,
    )
    prob_dead = mother_survival_prob[0]
    prob_alive = mother_survival_prob[1]

    outcome_bad_health = (
        options["mother_bad_health"]["bad_health_age"] * mother_age
        + options["mother_bad_health"]["bad_health_age_squared"] * mother_age_squared
        + options["mother_bad_health"]["bad_health_lagged_good_health"] * good_health
        + options["mother_bad_health"]["bad_health_lagged_bad_health"] * bad_health
        + options["mother_bad_health"]["bad_health_constant"]
    )

    linear_comb = jnp.array([0, outcome_bad_health])
    health_trans_probs = _softmax(linear_comb)

    return jnp.array(
        [
            health_trans_probs[0] * prob_alive,
            health_trans_probs[1] * prob_alive,
            prob_dead,
        ],
    )


def exog_health_transition_mother_with_survival_medium_bad(
    period,
    mother_health,
    options,
):
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
    mother_age = options["mother_start_age"] + period
    mother_age_squared = mother_age**2

    good_health = mother_health == GOOD_HEALTH
    medium_health = mother_health == MEDIUM_HEALTH
    bad_health = mother_health == BAD_HEALTH
    alive = mother_health != DEAD

    mother_survival_prob = prob_survival_mother(
        period=period,
        mother_health=mother_health,
        mother_alive=alive,
        options=options,
    )
    prob_dead = mother_survival_prob[0]
    prob_alive = mother_survival_prob[1]

    # Linear combination for medium health
    lc_medium_health = (
        options["mother_medium_health"]["medium_health_age"] * mother_age
        + options["mother_medium_health"]["medium_health_age_squared"]
        * mother_age_squared
        + options["mother_medium_health"]["medium_health_lagged_good_health"]
        * good_health
        + options["mother_medium_health"]["medium_health_lagged_medium_health"]
        * medium_health
        + options["mother_medium_health"]["medium_health_lagged_bad_health"]
        * bad_health
        + options["mother_medium_health"]["medium_health_constant"]
    )

    # Linear combination for bad health
    lc_bad_health = (
        options["mother_bad_health"]["bad_health_age"] * mother_age
        + options["mother_bad_health"]["bad_health_age_squared"] * mother_age_squared
        + options["mother_bad_health"]["bad_health_lagged_good_health"] * good_health
        + options["mother_bad_health"]["bad_health_lagged_medium_health"]
        * medium_health
        + options["mother_bad_health"]["bad_health_lagged_bad_health"] * bad_health
        + options["mother_bad_health"]["bad_health_constant"]
    )

    linear_comb = jnp.array([0, lc_medium_health, lc_bad_health])
    health_trans_probs = _softmax(linear_comb)

    return jnp.array(
        [
            health_trans_probs[0] * prob_alive,
            health_trans_probs[1] * prob_alive,
            health_trans_probs[2] * prob_alive,
            prob_dead,
        ],
    )


def exog_health_transition_mother(period, mother_health, options):
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
    mother_age = options["mother_start_age"] + period
    mother_age_squared = mother_age**2

    good_health = mother_health == GOOD_HEALTH
    medium_health = mother_health == MEDIUM_HEALTH
    bad_health = mother_health == BAD_HEALTH

    # Linear combination for medium health
    lc_medium_health = (
        options["mother_medium_health"]["medium_health_age"] * mother_age
        + options["mother_medium_health"]["medium_health_age_squared"]
        * mother_age_squared
        + options["mother_medium_health"]["medium_health_lagged_good_health"]
        * good_health
        + options["mother_medium_health"]["medium_health_lagged_medium_health"]
        * medium_health
        + options["mother_medium_health"]["medium_health_lagged_bad_health"]
        * bad_health
        + options["mother_medium_health"]["medium_health_constant"]
    )

    # Linear combination for bad health
    lc_bad_health = (
        options["mother_bad_health"]["bad_health_age"] * mother_age
        + options["mother_bad_health"]["bad_health_age_squared"] * mother_age_squared
        + options["mother_bad_health"]["bad_health_lagged_good_health"] * good_health
        + options["mother_bad_health"]["bad_health_lagged_medium_health"]
        * medium_health
        + options["mother_bad_health"]["bad_health_lagged_bad_health"] * bad_health
        + options["mother_bad_health"]["bad_health_constant"]
    )

    linear_comb = jnp.array([0, lc_medium_health, lc_bad_health])
    transition_probs = _softmax(linear_comb)

    return jnp.array([transition_probs[0], transition_probs[1], transition_probs[2]])


def _softmax(lc):
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
    e_lc = jnp.exp(lc - jnp.max(lc))  # Subtract max for numerical stability
    return e_lc / e_lc.sum(axis=0)


# =====================================================================================
# Care demand
# =====================================================================================


def prob_exog_care_demand(
    period,
    mother_health,
    mother_alive,
    options,
):
    """Create nested exogenous care demand probabilities.

    Compute based on parent alive. Otherwise zero.
    Done outside?!

    Nested exogenous transitions:
    - First, a parent's health state is determined by their age and lagged health state.

    Args:
        period (int): Current period.
        parental_age (int): Age of parent.
        parent_alive (int): Binary indicator of whether parent is alive.
        good_health (int): Binary indicator of good health.
        medium_health (int): Binary indicator of medium health.
        bad_health (int): Binary indicator of bad health.
        params (dict): Dictionary of parameters.
        mother_alive (int): Binary indicator of whether mother is alive.
        mother_health (int): Binary indicator of mother's health state.
        father_alive (int): Binary indicator of whether father is alive.
        father_health (int): Binary indicator of father's health state.
        options (dict): Dictionary of options.

    Returns:
        jnp.ndarray: Array of shape (2,) representing the probabilities of
            no care demand and care demand, respectively.

    """
    mother_survival_prob = prob_survival_mother(
        period=period,
        mother_health=mother_health,
        mother_alive=mother_alive,
        options=options,
    )

    mother_trans_probs_health = exog_health_transition_mother(
        period=period,
        mother_health=mother_health,
        options=options,
    )
    # ===============================================================

    mother_prob_care_good = _exog_care_demand_mother(
        period=period,
        mother_health=0,
        options=options,
    )
    mother_prob_care_medium = _exog_care_demand_mother(
        period=period,
        mother_health=1,
        options=options,
    )
    mother_prob_care_bad = _exog_care_demand_mother(
        period=period,
        mother_health=2,
        options=options,
    )

    _mother_trans_probs_care_demand = jnp.array(
        [mother_prob_care_good, mother_prob_care_medium, mother_prob_care_bad],
    )

    # Non-zero probability of care demand only if parent is alive,
    # weighted by the parent's survival probability
    prob_care_demand = (mother_survival_prob * mother_alive) * (
        mother_trans_probs_health @ _mother_trans_probs_care_demand
    )

    return jnp.array([1 - prob_care_demand, prob_care_demand])


def _exog_care_demand_mother(period, mother_health, options):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    mother_age = options["mother_start_age"] + period

    logit = (
        options["exog_care_single_mother_constant"]
        + options["exog_care_single_mother_age"] * mother_age
        + options["exog_care_single_mother_age_squared"] * (mother_age**2)
        + options["exog_care_single_mother_health_medium"]
        * (mother_health == MEDIUM_HEALTH)
        + options["exog_care_single_mother_health_bad"] * (mother_health == BAD_HEALTH)
    )
    return 1 / (1 + jnp.exp(-logit))


def _exog_care_demand_father(period, father_health, options):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    father_age = options["father_start_age"] + period

    logit = (
        options["exog_care_single_father_constant"]
        + options["exog_care_single_father_age"] * father_age
        + options["exog_care_single_father_age_squared"] * (father_age**2)
        + options["exog_care_single_father_health_medium"]
        * (father_health == MEDIUM_HEALTH)
        + options["exog_care_single_father_health_bad"] * (father_health == BAD_HEALTH)
    )
    return 1 / (1 + jnp.exp(-logit))


def _exog_care_demand_couple(period, mother_health, father_health, options):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    mother_age = options["mother_start_age"] + period
    father_age = options["father_start_age"] + period

    logit = (
        options["exog_care_couple_constant"]
        + options["exog_care_couple_mother_age"] * mother_age
        + options["exog_care_couple_mother_age_squared"] * (mother_age**2)
        + options["exog_care_couple_mother_medium_health"]
        * (mother_health == MEDIUM_HEALTH)
        + options["exog_care_couple_mother_bad_health"] * (mother_health == BAD_HEALTH)
        + options["exog_care_couple_father_age"] * father_age
        + options["exog_care_couple_father_age_squared"] * (father_age**2)
        + options["exog_care_couple_father_medium_health"]
        * (father_health == MEDIUM_HEALTH)
        + options["exog_care_couple_father_bad_health"] * (father_health == BAD_HEALTH)
    )
    return 1 / (1 + jnp.exp(-logit))
