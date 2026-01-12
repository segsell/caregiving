"""Job finding and job separation processes."""

import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
    UNEMPLOYED_CHOICES,
    WORK_CHOICES,
    is_retired,
    is_unemployed,
    is_working,
)

AGE_THRESHOLD_50 = 50
AGE_THRESHOLD_55 = 55
AGE_THRESHOLD_60 = 60


def job_offer_process_transition(params, model_specs, education, period, choice):
    """Transition probability for next period job offer state.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    """

    retirement_choice = is_retired(choice)
    unemployment_choice = is_unemployed(choice)
    labor_choice = is_working(choice)

    job_sep_prob = model_specs["job_sep_probs"][SEX, education, period]

    job_finding_prob = calc_job_finding_prob_women(
        period, education, params, model_specs
    )

    # Transition probability
    prob_no_job = (
        job_sep_prob * labor_choice
        + (1 - job_finding_prob) * unemployment_choice
        + 1 * retirement_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])


def job_offer_process_transition_initial_conditions(
    params, model_specs, education, period, choice
):
    """Transition probability for next period job offer state.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    """

    unemployment_choice = jnp.isin(choice, UNEMPLOYED_CHOICES)
    labor_choice = jnp.isin(choice, WORK_CHOICES)

    job_sep_prob = model_specs["job_sep_probs"][SEX, education, period]

    job_finding_prob = calc_job_finding_prob_women_age_dummies(
        period, education, params, model_specs
    )

    # Transition probability
    prob_no_job = (
        job_sep_prob * labor_choice + (1 - job_finding_prob) * unemployment_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])


def calc_job_finding_prob_women(period, education, params, model_specs):
    high_edu = education == 1
    age = period + model_specs["start_age"]

    exp_factor = (
        params["job_finding_logit_const_women"]
        + params["job_finding_logit_age_women"] * age
        + params["job_finding_logit_high_educ_women"] * high_edu
    )
    prob = logit_formula(exp_factor)

    return prob


def calc_job_finding_prob_women_age_dummies(period, education, params, model_specs):
    age = period + model_specs["start_age"]
    high_edu = education == 1

    above_55 = age >= AGE_THRESHOLD_55
    above_50 = age >= AGE_THRESHOLD_50
    above_60 = age >= AGE_THRESHOLD_60

    exp_factor = (
        params["job_finding_logit_const_women"]
        + params["job_finding_logit_high_educ_women"] * high_edu
        + params["job_finding_logit_above_50_women"] * above_50
        + params["job_finding_logit_above_55_women"] * above_55
        + params["job_finding_logit_above_60_women"] * above_60
    )
    prob = logit_formula(exp_factor)

    return prob


def logit_formula(x):
    return 1 / (1 + jnp.exp(-x))
