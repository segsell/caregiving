"""Job finding and job separation processes for job retention counterfactual.

This counterfactual implements a policy where caregivers can keep their jobs
even when they reduce hours or become unemployed due to caregiving activities.
"""

import jax.numpy as jnp

from caregiving.model.shared import (
    SEX,
    had_ft_job_before_caregiving,
    had_job_before_caregiving,
    had_pt_job_before_caregiving,
    is_full_time,
    is_informal_care,
    is_part_time,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.stochastic_processes.job_transition import (
    calc_job_finding_prob_women_age_dummies,
)


def job_offer_process_transition_with_job_retention(
    params, model_specs, education, period, choice, job_before_caregiving
):
    """Transition probability for next period job offer state with job retention policy.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    Job Retention Policy:
    - For people who are caregiving AND had a job before caregiving started,
      job_sep_prob = 0
    - For people who are caregiving AND had a job before caregiving AND
      on "caregiving leave" (unemployed or part-time):
      job_finding_prob = 1.0 (guaranteed job offer after caregiving ends)
    - For people who are caregiving AND had a job before caregiving AND
      working full-time: job_finding_prob = normal probability
      (no special benefit, treated like non-caregiving)
    - Retired caregivers are excluded from the job retention benefit because
      retirement already sets job offer probability to 0 in the final calculation
    - This allows caregivers on leave to keep their previous job and return to it
      after caregiving ends, but full-time workers receive normal job offer probability

    """

    retirement_choice = is_retired(choice)
    unemployment_choice = is_unemployed(choice)
    labor_choice = is_working(choice)

    caregiving_choice = is_informal_care(choice)
    employed_before_caregiving = had_job_before_caregiving(job_before_caregiving)

    job_sep_prob = model_specs["job_sep_probs"][SEX, education, period]
    job_finding_prob = calc_job_finding_prob_women_age_dummies(
        period, education, params, model_specs
    )

    # Job separation policy: if caregiving + had job before caregiving,
    # no job separation
    caregiver_with_previous_job = employed_before_caregiving & caregiving_choice
    job_sep_prob_with_caregiver_retention = jnp.where(
        caregiver_with_previous_job, 0.0, job_sep_prob
    )

    # Check if person is on "caregiving leave"
    # (unemployed or part-time while caregiving)
    # Note: Retired caregivers are excluded because retirement already sets
    # prob_no_job = 1 (prob_job = 0) in the final calculation, so the job
    # retention benefit doesn't apply
    caregiving_leave = caregiving_choice & (
        is_unemployed(choice) | is_part_time(choice) | is_full_time(choice)
    )

    # Job finding probability modification:
    # - If caregiving + had job before + on caregiving leave (unemployed/part-time):
    #   guaranteed job offer (prob = 1.0)
    # - If caregiving + had job before + full-time: normal job finding probability
    # - If caregiving + had job before + retired: normal job finding probability
    #   (but retirement term in prob_no_job calculation will override to prob_job = 0)
    # - Otherwise: use normal job finding probability
    job_finding_prob_with_caregiver_retention = jnp.where(
        caregiving_leave & employed_before_caregiving,
        1.0,  # Guaranteed job offer for caregiving leave
        job_finding_prob,  # Normal job finding probability
    )

    # Transition probability
    prob_no_job = (
        job_sep_prob_with_caregiver_retention * labor_choice
        + (1 - job_finding_prob_with_caregiver_retention) * unemployment_choice
        + 1 * retirement_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])


def job_offer_process_transition_leave_with_job_retention(
    params, model_specs, education, period, choice, job_before_caregiving
):
    """Transition probability for next period job offer state with job retention policy.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    Job Retention Policy:
    - For people who are caregiving AND had a job before caregiving started,
      job_sep_prob = 0
    - For people who are caregiving AND had a job before caregiving AND
      on "caregiving leave" (unemployed or part-time):
      job_finding_prob = 1.0 (guaranteed job offer after caregiving ends)
    - For people who are caregiving AND had a job before caregiving AND
      working full-time: job_finding_prob = normal probability
      (no special benefit, treated like non-caregiving)
    - Retired caregivers are excluded from the job retention benefit because
      retirement already sets job offer probability to 0 in the final calculation
    - This allows caregivers on leave to keep their previous job and return to it
      after caregiving ends, but full-time workers receive normal job offer probability

    """

    retirement_choice = is_retired(choice)
    unemployment_choice = is_unemployed(choice)
    labor_choice = is_working(choice)

    caregiving_choice = is_informal_care(choice)
    employed_before_caregiving = had_pt_job_before_caregiving(
        job_before_caregiving
    ) | had_ft_job_before_caregiving(job_before_caregiving)

    job_sep_prob = model_specs["job_sep_probs"][SEX, education, period]
    job_finding_prob = calc_job_finding_prob_women_age_dummies(
        period, education, params, model_specs
    )

    # Job separation policy: if caregiving + had job before caregiving,
    # no job separation
    caregiver_with_previous_job = employed_before_caregiving & caregiving_choice
    job_sep_prob_with_caregiver_retention = jnp.where(
        caregiver_with_previous_job, 0.0, job_sep_prob
    )

    # Check if person is on "caregiving leave"
    # (unemployed or part-time while caregiving)
    # Note: Retired caregivers are excluded because retirement already sets
    # prob_no_job = 1 (prob_job = 0) in the final calculation, so the job
    # retention benefit doesn't apply
    caregiving_leave = caregiving_choice & (
        is_unemployed(choice) | is_part_time(choice) | is_full_time(choice)
    )

    # Job finding probability modification:
    # - If caregiving + had job before + on caregiving leave (unemployed/part-time):
    #   guaranteed job offer (prob = 1.0)
    # - If caregiving + had job before + full-time: normal job finding probability
    # - If caregiving + had job before + retired: normal job finding probability
    #   (but retirement term in prob_no_job calculation will override to prob_job = 0)
    # - Otherwise: use normal job finding probability
    job_finding_prob_with_caregiver_retention = jnp.where(
        caregiving_leave & employed_before_caregiving,
        1.0,  # Guaranteed job offer for caregiving leave
        job_finding_prob,  # Normal job finding probability
    )

    # Transition probability
    prob_no_job = (
        job_sep_prob_with_caregiver_retention * labor_choice
        + (1 - job_finding_prob_with_caregiver_retention) * unemployment_choice
        + 1 * retirement_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])
