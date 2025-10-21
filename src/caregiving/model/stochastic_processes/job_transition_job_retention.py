"""Job finding and job separation processes for job retention counterfactual.

This counterfactual implements a policy where caregivers can keep their jobs
even when they reduce hours or become unemployed due to caregiving activities.
"""

import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
    had_job_before_caregiving,
    is_informal_care,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.stochastic_processes.job_transition import (
    calc_job_finding_prob_women,
)


def job_offer_process_transition_with_job_retention(
    params, options, education, period, choice, job_before_caregiving
):
    """Transition probability for next period job offer state with job retention policy.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    Job Retention Policy:
    - For people who are caregiving AND had a job before caregiving started,
      job_sep_prob = 0
    - For people who are caregiving AND had a job before caregiving,
      job_finding_prob is increased
    - This allows caregivers to keep their previous job and return to it
      after caregiving ends

    """

    retirement_choice = is_retired(choice)
    unemployment_choice = is_unemployed(choice)
    labor_choice = is_working(choice)

    caregiving_choice = is_informal_care(choice)
    employed_before_caregiving = had_job_before_caregiving(job_before_caregiving)

    # Job separation policy: if caregiving + had job before caregiving,
    # no job separation
    labor_and_caregiving_with_previous_job = (
        employed_before_caregiving & caregiving_choice
    )

    job_sep_prob = options["job_sep_probs"][SEX, education, period]
    job_finding_prob = calc_job_finding_prob_women(period, education, params, options)

    # Apply job retention policy: if caregiving + had job before caregiving,
    # no job separation
    job_sep_prob_with_caregiver_retention = jnp.where(
        labor_and_caregiving_with_previous_job, 0.0, job_sep_prob
    )

    job_finding_prob_with_caregiver_retention = jnp.where(
        labor_and_caregiving_with_previous_job,
        1.0,
        job_finding_prob,
    )

    # Transition probability
    prob_no_job = (
        job_sep_prob_with_caregiver_retention * labor_choice
        + (1 - job_finding_prob_with_caregiver_retention) * unemployment_choice
        + 1 * retirement_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])
