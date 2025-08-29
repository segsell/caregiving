import pickle as pkl
from itertools import product

import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import WORK_NO_CARE
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition,
)

EDU_GRID = [0, 1]
PERIOD_GRID = np.arange(0, 20, 2, dtype=int)
LOGIT_PARAM_GRID = np.arange(0.1, 0.9, 0.2)
WORK_CHOICE_GRID = WORK_NO_CARE
SEX_GRID = [1]


@pytest.fixture(scope="module")
def load_specs():
    """Load specs from pickle file."""

    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    return specs


@pytest.mark.parametrize(
    "education, sex, period, logit_param, work_choice",
    list(product(EDU_GRID, SEX_GRID, PERIOD_GRID, LOGIT_PARAM_GRID, WORK_CHOICE_GRID)),
)
def test_job_destruction(education, sex, period, logit_param, work_choice, load_specs):
    """Test job destruction probs."""
    options = load_specs

    params = {}
    append = "women"
    gender_params = {
        f"job_finding_logit_const_{append}": logit_param,
        f"job_finding_logit_age_{append}": logit_param,
        f"job_finding_logit_high_educ_{append}": logit_param,
        f"job_finding_logit_age_squared_{append}": logit_param,
        f"job_finding_logit_age_cubed_{append}": logit_param,
    }
    params = {**params, **gender_params}
    job_dest_prob = options["job_sep_probs"][sex, education, period]
    full_probs_expec = np.array([job_dest_prob, 1 - job_dest_prob])

    probs = job_offer_process_transition(
        params=params,
        options=options,
        education=education,
        # sex=sex,
        period=period,
        choice=work_choice,
    )

    np.testing.assert_almost_equal(probs, full_probs_expec)
    np.testing.assert_allclose(probs.sum(), 1.0)
