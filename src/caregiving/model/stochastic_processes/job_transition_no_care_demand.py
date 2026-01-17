"""Job finding and job separation processes for no-care-demand counterfactual."""

import jax.numpy as jnp

from caregiving.model.shared import SEX
from caregiving.model.shared_no_care_demand import (
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.stochastic_processes.job_transition import (
    calc_job_finding_prob_women_linear,
)


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

    job_finding_prob = calc_job_finding_prob_women_linear(
        period, education, params, model_specs
    )

    # Transition probability
    prob_no_job = (
        job_sep_prob * labor_choice
        + (1 - job_finding_prob) * unemployment_choice
        + 1 * retirement_choice
    )

    return jnp.array([prob_no_job, 1 - prob_no_job])
