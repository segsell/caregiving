import pickle as pkl
from itertools import product

import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.stochastic_processes.partner_transition import partner_transition

EDU_GRID = np.arange(2, dtype=int)
PERIOD_GRID = np.linspace(0, 45, 1, dtype=int)
PARTNER_STATE_GRID = np.arange(3, dtype=int)
SEX_GRID = [1]


@pytest.fixture(scope="module")
def load_specs():
    """Load specs from pickle file."""

    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    return specs


@pytest.mark.parametrize(
    "education, sex, period, partner_state",
    list(product(EDU_GRID, SEX_GRID, PERIOD_GRID, PARTNER_STATE_GRID)),
)
def test_vec_shape(education, sex, period, partner_state, load_specs):
    """Test shape of transition vector."""
    specs = load_specs

    res = partner_transition(
        period=period,
        education=education,
        # sex=sex,
        partner_state=partner_state,
        options=specs,
    )

    assert res.shape == (specs["n_partner_states"],)
    np.testing.assert_allclose(res.sum(), 1.0)
