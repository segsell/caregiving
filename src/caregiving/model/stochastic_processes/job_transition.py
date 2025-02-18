"""Job finding and job separation processes."""

import jax.numpy as jnp

from caregiving.model.shared import SEX, is_unemployed, is_working


def job_offer_process_transition(params, options, education, period, choice):
    """Transition probability for next period job offer state.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    """

    unemployment_choice = is_unemployed(choice)
    labor_choice = is_working(choice)

    job_sep_prob = options["job_sep_probs"][SEX, education, period]

    job_finding_prob = calc_job_finding_prob_women(period, education, params, options)

    # Transition probability
    prob_no_job = (
        job_sep_prob * labor_choice + (1 - job_finding_prob) * unemployment_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])


def calc_job_finding_prob_women(period, education, params, options):
    high_edu = education == 1
    age = period + options["start_age"]

    # above_49 = age > 49
    exp_value = jnp.exp(
        params["job_finding_logit_const_women"]
        + params["job_finding_logit_age_women"] * age
        # + params["job_finding_logit_above_49"] * above_49
        + params["job_finding_logit_high_educ_women"] * high_edu
    )

    return exp_value / (1 + exp_value)
