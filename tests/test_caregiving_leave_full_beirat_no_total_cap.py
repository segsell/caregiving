"""Tests for the Full-Beirat-no-total-cap caregiving leave counterfactual.

The variant is a near-fork of Full Beirat with two precise changes:
- ``years_leave_used_total`` is dropped from the deterministic states.
- partial leave (PT with prior FT) is uncapped; full leave (unemployed with
  prior job) is still capped at one year via ``full_leave_year_used``.

These tests pair the new variant against the Full Beirat benchmark using:
1. Equivalence at ``years_leave_used_total == 0`` (no Full Beirat cap binds).
2. Differential at ``years_leave_used_total == LEAVE_CAP_YEARS`` (proves the
   total cap was actually removed and the full-leave sub-cap is still
   enforced).
3. Structural invariants on the state transition, sparsity, and factory.
"""

import pickle as pkl
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.experience_caregiving_leave_model import (
    get_next_period_experience_caregiving_leave_full_beirat,
    get_next_period_experience_caregiving_leave_full_beirat_no_total_cap,
)
from caregiving.model.shared import (
    DEAD,
    GOOD_HEALTH,
    JOB_RETENTION_FULL_TIME,
    JOB_RETENTION_PART_TIME,
    LEAVE_CAP_YEARS,
    NO_CARE_DEMAND,
    PARENT_LONGER_DEAD,
)
from caregiving.model.state_space import (
    state_specific_choice_set_with_caregiving,
)
from caregiving.model.state_space_caregiving_leave_beirat import (
    create_state_space_functions_full_beirat_no_total_cap,
    next_period_deterministic_state_full_beirat,
    next_period_deterministic_state_full_beirat_no_total_cap,
    sparsity_condition_full_beirat_no_total_cap,
)
from caregiving.model.wealth_and_budget.caregiving_leave_top_up import (
    calc_caregiving_leave_top_up_full_beirat,
    calc_caregiving_leave_top_up_full_beirat_no_total_cap,
)

jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------------------------
# Choice integers (see src/caregiving/model/shared.py for the encoding).
#
# Labor state x care arrangement:
#   0  retired,        no care            8  retired,        light informal
#   1  unemployed,     no care            9  unemployed,     light informal
#   2  part-time,      no care           10  part-time,      light informal
#   3  full-time,      no care           11  full-time,      light informal
#   4  retired,        formal            12  retired,        intensive informal
#   5  unemployed,     formal            13  unemployed,     intensive informal
#   6  part-time,      formal            14  part-time,      intensive informal
#   7  full-time,      formal            15  full-time,      intensive informal
#
# is_informal_care True iff choice >= 8. So an agent on "full leave" (unemployed
# AND providing informal care) must use choice 9 or 13, NOT 5 (which is formal
# care).  Likewise "partial leave" (PT with prior FT, providing informal care)
# requires choice 10 or 14.
# -------------------------------------------------------------------------------------
RETIRED_NO_CARE = 0
UNEMP_NO_CARE = 1
PT_NO_CARE = 2
FT_NO_CARE = 3
UNEMP_FORMAL_CARE = 5
UNEMP_LIGHT_INFORMAL = 9
PT_LIGHT_INFORMAL = 10
UNEMP_INTENSIVE_INFORMAL = 13
PT_INTENSIVE_INFORMAL = 14


@pytest.fixture(scope="module")
def load_specs():
    """Load specs from the on-disk pickle (same convention as other tests)."""
    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"
    with path_to_specs.open("rb") as file:
        return pkl.load(file)


# Default kwargs that don't drive caps/eligibility but are required by the API.
DEFAULT_EXPERIENCE_YEARS = 10.0
DEFAULT_INCOME_SHOCK = 0.0
DEFAULT_SEX = 1
DEFAULT_LABOR_INCOME = 0.0  # keeps the prior-FT-to-PT top-up strictly positive


# =====================================================================================
# Group 1: Equivalence between new variant and Full Beirat at years_leave_used_total=0
# =====================================================================================

EQUIV_CHOICES = [
    RETIRED_NO_CARE,
    UNEMP_NO_CARE,
    FT_NO_CARE,
    UNEMP_FORMAL_CARE,  # is_unemployed but not is_informal_care -> top-up=0
    UNEMP_LIGHT_INFORMAL,  # full-leave eligible if prior job
    UNEMP_INTENSIVE_INFORMAL,  # full-leave eligible if prior job
    PT_LIGHT_INFORMAL,  # partial-leave eligible if prior FT
    PT_INTENSIVE_INFORMAL,
]
EQUIV_JOB_BEFORE = [0, JOB_RETENTION_PART_TIME, JOB_RETENTION_FULL_TIME]
EQUIV_FULL_USED = [0, 1]
EQUIV_EDUCATION = [0, 1]


@pytest.mark.parametrize(
    "lagged_choice, job_before_caregiving, full_leave_year_used, education",
    list(product(EQUIV_CHOICES, EQUIV_JOB_BEFORE, EQUIV_FULL_USED, EQUIV_EDUCATION)),
)
def test_topup_equivalence_at_total_zero(
    lagged_choice,
    job_before_caregiving,
    full_leave_year_used,
    education,
    load_specs,
):
    """At years_leave_used_total=0 the no-total-cap top-up must equal the FB one."""
    specs = load_specs

    topup_new = calc_caregiving_leave_top_up_full_beirat_no_total_cap(
        lagged_choice=lagged_choice,
        education=education,
        job_before_caregiving=job_before_caregiving,
        experience_years=DEFAULT_EXPERIENCE_YEARS,
        income_shock_previous_period=DEFAULT_INCOME_SHOCK,
        sex=DEFAULT_SEX,
        labor_income_after_ssc=DEFAULT_LABOR_INCOME,
        full_leave_year_used=full_leave_year_used,
        model_specs=specs,
    )
    topup_old = calc_caregiving_leave_top_up_full_beirat(
        lagged_choice=lagged_choice,
        education=education,
        job_before_caregiving=job_before_caregiving,
        experience_years=DEFAULT_EXPERIENCE_YEARS,
        income_shock_previous_period=DEFAULT_INCOME_SHOCK,
        sex=DEFAULT_SEX,
        labor_income_after_ssc=DEFAULT_LABOR_INCOME,
        years_leave_used_total=0,
        full_leave_year_used=full_leave_year_used,
        model_specs=specs,
    )

    np.testing.assert_allclose(
        np.asarray(topup_new), np.asarray(topup_old), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "lagged_choice, job_before_caregiving, full_leave_year_used, education",
    list(product(EQUIV_CHOICES, EQUIV_JOB_BEFORE, EQUIV_FULL_USED, EQUIV_EDUCATION)),
)
def test_experience_equivalence_at_total_zero(
    lagged_choice,
    job_before_caregiving,
    full_leave_year_used,
    education,
    load_specs,
):
    """At years_leave_used_total=0 the no-total-cap experience update equals FB."""
    specs = load_specs
    period = 20  # age 50, mid-career, well within caregiving window
    already_retired = 0
    # partner_state must be a jax-y scalar because the experience pipeline
    # calls (partner_state > 0).astype(int) downstream.
    partner_state = jnp.asarray(0)
    experience = 0.5

    exp_new = get_next_period_experience_caregiving_leave_full_beirat_no_total_cap(
        period=period,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        partner_state=partner_state,
        education=education,
        experience=experience,
        job_before_caregiving=job_before_caregiving,
        full_leave_year_used=full_leave_year_used,
        model_specs=specs,
    )
    exp_old = get_next_period_experience_caregiving_leave_full_beirat(
        period=period,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        partner_state=partner_state,
        education=education,
        experience=experience,
        job_before_caregiving=job_before_caregiving,
        years_leave_used_total=0,
        full_leave_year_used=full_leave_year_used,
        model_specs=specs,
    )

    np.testing.assert_allclose(np.asarray(exp_new), np.asarray(exp_old), rtol=0, atol=0)


@pytest.mark.parametrize(
    "choice, lagged_choice, job_before_caregiving, full_leave_year_used",
    list(
        product(
            EQUIV_CHOICES,
            EQUIV_CHOICES,
            EQUIV_JOB_BEFORE,
            EQUIV_FULL_USED,
        )
    ),
)
def test_state_transition_equivalence_at_total_zero(
    choice,
    lagged_choice,
    job_before_caregiving,
    full_leave_year_used,
):
    """At years_leave_used_total=0 the FB transition must agree on shared keys.

    The new transition drops ``years_leave_used_total``; we therefore compare
    only on the keys both functions return.
    """
    period = 20
    already_retired = 0

    out_new = next_period_deterministic_state_full_beirat_no_total_cap(
        period=period,
        choice=choice,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        job_before_caregiving=job_before_caregiving,
        full_leave_year_used=full_leave_year_used,
    )
    out_old = next_period_deterministic_state_full_beirat(
        period=period,
        choice=choice,
        lagged_choice=lagged_choice,
        already_retired=already_retired,
        job_before_caregiving=job_before_caregiving,
        years_leave_used_total=0,
        full_leave_year_used=full_leave_year_used,
    )

    shared_keys = set(out_new.keys()) & set(out_old.keys())
    assert shared_keys == {
        "period",
        "lagged_choice",
        "already_retired",
        "job_before_caregiving",
        "full_leave_year_used",
    }
    for key in shared_keys:
        np.testing.assert_allclose(
            np.asarray(out_new[key]), np.asarray(out_old[key]), rtol=0, atol=0
        )


# =====================================================================================
# Group 2: Differential -- partial leave is uncapped in the new variant
# =====================================================================================


@pytest.mark.parametrize("lagged_choice", [PT_LIGHT_INFORMAL, PT_INTENSIVE_INFORMAL])
def test_topup_partial_leave_uncapped_when_full_beirat_would_be_capped(
    lagged_choice, load_specs
):
    """PT-with-prior-FT caregiver: capped under FB at total=LEAVE_CAP_YEARS, NOT
    capped under the no-total-cap variant."""
    specs = load_specs

    topup_old_capped = calc_caregiving_leave_top_up_full_beirat(
        lagged_choice=lagged_choice,
        education=0,
        job_before_caregiving=JOB_RETENTION_FULL_TIME,
        experience_years=DEFAULT_EXPERIENCE_YEARS,
        income_shock_previous_period=DEFAULT_INCOME_SHOCK,
        sex=DEFAULT_SEX,
        labor_income_after_ssc=DEFAULT_LABOR_INCOME,
        years_leave_used_total=LEAVE_CAP_YEARS,
        full_leave_year_used=0,
        model_specs=specs,
    )
    topup_new_uncapped = calc_caregiving_leave_top_up_full_beirat_no_total_cap(
        lagged_choice=lagged_choice,
        education=0,
        job_before_caregiving=JOB_RETENTION_FULL_TIME,
        experience_years=DEFAULT_EXPERIENCE_YEARS,
        income_shock_previous_period=DEFAULT_INCOME_SHOCK,
        sex=DEFAULT_SEX,
        labor_income_after_ssc=DEFAULT_LABOR_INCOME,
        full_leave_year_used=0,
        model_specs=specs,
    )

    assert float(topup_old_capped) == 0.0
    assert float(topup_new_uncapped) > 0.0


@pytest.mark.parametrize("lagged_choice", [PT_LIGHT_INFORMAL])
def test_experience_partial_leave_frozen_when_full_beirat_would_unfreeze(
    lagged_choice, load_specs
):
    """At FB's total cap, partial-leave experience grows at baseline PT rate;
    in the no-total-cap variant it stays frozen at prior-FT (=1.0).

    Note: only PT_LIGHT_INFORMAL exercises the differential. For
    PT_INTENSIVE_INFORMAL, the baseline experience update is already 1.0
    (intensive carers receive FT-equivalent credit while PT, see
    ``exp_update_baseline`` in the experience function), so freeze-vs-no-freeze
    produces the same update by construction and the differential is invisible
    there.
    """
    specs = load_specs
    period = 20

    partner_state = jnp.asarray(0)
    exp_old_capped = get_next_period_experience_caregiving_leave_full_beirat(
        period=period,
        lagged_choice=lagged_choice,
        already_retired=0,
        partner_state=partner_state,
        education=0,
        experience=0.5,
        job_before_caregiving=JOB_RETENTION_FULL_TIME,
        years_leave_used_total=LEAVE_CAP_YEARS,
        full_leave_year_used=0,
        model_specs=specs,
    )
    exp_new_uncapped = (
        get_next_period_experience_caregiving_leave_full_beirat_no_total_cap(
            period=period,
            lagged_choice=lagged_choice,
            already_retired=0,
            partner_state=partner_state,
            education=0,
            experience=0.5,
            job_before_caregiving=JOB_RETENTION_FULL_TIME,
            full_leave_year_used=0,
            model_specs=specs,
        )
    )

    # No-total-cap freezes at the FT path (+1.0); FB at the cap uses the
    # baseline PT rate (< 1.0). Hence the new variant must give a strictly
    # larger experience update.
    assert float(exp_new_uncapped) > float(exp_old_capped)


# =====================================================================================
# Group 3: Differential -- 1-year full-leave sub-cap is still enforced
# =====================================================================================


@pytest.mark.parametrize(
    "lagged_choice, prior_job",
    list(
        product(
            [UNEMP_LIGHT_INFORMAL, UNEMP_INTENSIVE_INFORMAL],
            [JOB_RETENTION_PART_TIME, JOB_RETENTION_FULL_TIME],
        )
    ),
)
def test_topup_full_leave_still_capped_at_full_leave_year_used_one(
    lagged_choice, prior_job, load_specs
):
    """Full leave (unemployed + caregiving + prior job) gives 0 top-up when
    full_leave_year_used==1, in BOTH the new variant and Full Beirat."""
    specs = load_specs

    topup_new = calc_caregiving_leave_top_up_full_beirat_no_total_cap(
        lagged_choice=lagged_choice,
        education=0,
        job_before_caregiving=prior_job,
        experience_years=DEFAULT_EXPERIENCE_YEARS,
        income_shock_previous_period=DEFAULT_INCOME_SHOCK,
        sex=DEFAULT_SEX,
        labor_income_after_ssc=DEFAULT_LABOR_INCOME,
        full_leave_year_used=1,
        model_specs=specs,
    )
    topup_old = calc_caregiving_leave_top_up_full_beirat(
        lagged_choice=lagged_choice,
        education=0,
        job_before_caregiving=prior_job,
        experience_years=DEFAULT_EXPERIENCE_YEARS,
        income_shock_previous_period=DEFAULT_INCOME_SHOCK,
        sex=DEFAULT_SEX,
        labor_income_after_ssc=DEFAULT_LABOR_INCOME,
        years_leave_used_total=1,
        full_leave_year_used=1,
        model_specs=specs,
    )

    assert float(topup_new) == 0.0
    assert float(topup_old) == 0.0


def test_state_transition_full_leave_year_used_does_not_flip_after_one():
    """An unemp-care choice with full_leave_year_used=1 keeps it at 1 (monotone)."""
    out = next_period_deterministic_state_full_beirat_no_total_cap(
        period=20,
        choice=UNEMP_LIGHT_INFORMAL,
        lagged_choice=UNEMP_LIGHT_INFORMAL,
        already_retired=0,
        job_before_caregiving=JOB_RETENTION_FULL_TIME,
        full_leave_year_used=1,
    )
    assert int(np.asarray(out["full_leave_year_used"])) == 1


# =====================================================================================
# Group 4: State-transition invariants
# =====================================================================================


MONOTONE_CHOICES = [
    RETIRED_NO_CARE,
    UNEMP_NO_CARE,
    PT_NO_CARE,
    FT_NO_CARE,
    UNEMP_FORMAL_CARE,
    UNEMP_LIGHT_INFORMAL,
    UNEMP_INTENSIVE_INFORMAL,
    PT_LIGHT_INFORMAL,
    PT_INTENSIVE_INFORMAL,
]


@pytest.mark.parametrize(
    "choice, lagged_choice, job_before_caregiving, full_leave_year_used_in",
    list(
        product(
            MONOTONE_CHOICES,
            [UNEMP_NO_CARE, UNEMP_LIGHT_INFORMAL, FT_NO_CARE],
            EQUIV_JOB_BEFORE,
            EQUIV_FULL_USED,
        )
    ),
)
def test_full_leave_year_used_monotonicity(
    choice, lagged_choice, job_before_caregiving, full_leave_year_used_in
):
    """``full_leave_year_used`` is monotone non-decreasing."""
    out = next_period_deterministic_state_full_beirat_no_total_cap(
        period=20,
        choice=choice,
        lagged_choice=lagged_choice,
        already_retired=0,
        job_before_caregiving=job_before_caregiving,
        full_leave_year_used=full_leave_year_used_in,
    )
    assert int(np.asarray(out["full_leave_year_used"])) >= full_leave_year_used_in


@pytest.mark.parametrize(
    "choice, job_before_caregiving, expected_flip",
    [
        # Flips: full leave = unemployed AND informal care AND had prior job.
        (UNEMP_LIGHT_INFORMAL, JOB_RETENTION_PART_TIME, True),
        (UNEMP_LIGHT_INFORMAL, JOB_RETENTION_FULL_TIME, True),
        (UNEMP_INTENSIVE_INFORMAL, JOB_RETENTION_PART_TIME, True),
        (UNEMP_INTENSIVE_INFORMAL, JOB_RETENTION_FULL_TIME, True),
        # No flip: no prior job.
        (UNEMP_LIGHT_INFORMAL, 0, False),
        (UNEMP_INTENSIVE_INFORMAL, 0, False),
        # No flip: formal care (not informal).
        (UNEMP_FORMAL_CARE, JOB_RETENTION_FULL_TIME, False),
        # No flip: PT-while-caregiving is partial leave, not full leave.
        (PT_LIGHT_INFORMAL, JOB_RETENTION_FULL_TIME, False),
        (PT_INTENSIVE_INFORMAL, JOB_RETENTION_FULL_TIME, False),
        # No flip: no caregiving at all.
        (UNEMP_NO_CARE, JOB_RETENTION_FULL_TIME, False),
        (FT_NO_CARE, JOB_RETENTION_FULL_TIME, False),
        # No flip: retired.
        (RETIRED_NO_CARE, JOB_RETENTION_FULL_TIME, False),
    ],
)
def test_full_leave_year_used_flips_only_on_full_leave_choice(
    choice, job_before_caregiving, expected_flip
):
    """Starting from 0, flips to 1 iff the gates for full leave are satisfied."""
    out = next_period_deterministic_state_full_beirat_no_total_cap(
        period=20,
        choice=choice,
        lagged_choice=UNEMP_LIGHT_INFORMAL,
        already_retired=0,
        job_before_caregiving=job_before_caregiving,
        full_leave_year_used=0,
    )
    expected = 1 if expected_flip else 0
    assert int(np.asarray(out["full_leave_year_used"])) == expected


def test_no_years_leave_used_total_in_returned_dict():
    """The new transition must NOT leak the dropped state variable into outputs."""
    out = next_period_deterministic_state_full_beirat_no_total_cap(
        period=20,
        choice=UNEMP_LIGHT_INFORMAL,
        lagged_choice=UNEMP_LIGHT_INFORMAL,
        already_retired=0,
        job_before_caregiving=JOB_RETENTION_FULL_TIME,
        full_leave_year_used=0,
    )
    assert "years_leave_used_total" not in out
    assert "full_leave_year_used" in out
    # Base keys from next_period_deterministic_state_with_job_retention:
    for key in ("period", "lagged_choice", "already_retired", "job_before_caregiving"):
        assert key in out


# =====================================================================================
# Group 5: Sparsity invariants
# =====================================================================================


@pytest.mark.parametrize(
    "period, lagged_choice",
    [
        (15, UNEMP_NO_CARE),  # age 45, mid-career
        (20, UNEMP_NO_CARE),  # age 50
        (25, UNEMP_NO_CARE),  # age 55
    ],
)
def test_sparsity_rejects_non_caregiver_with_full_leave_used(
    period, lagged_choice, load_specs
):
    """caregiving_type=0 with full_leave_year_used=1 is impossible -> False."""
    specs = load_specs
    result = sparsity_condition_full_beirat_no_total_cap(
        period=period,
        lagged_choice=lagged_choice,
        already_retired=0,
        education=0,
        health=GOOD_HEALTH,
        partner_state=0,
        mother_adl=0,
        mother_dead=0,
        care_demand=NO_CARE_DEMAND,
        job_before_caregiving=0,
        full_leave_year_used=1,
        job_offer=1,
        caregiving_type=0,
        model_specs=specs,
    )
    assert result is False


@pytest.mark.parametrize(
    "full_leave_year_used",
    [0, 1],
)
def test_sparsity_accepts_caregiver_with_either_full_value(
    full_leave_year_used, load_specs
):
    """In an interior state, sparsity must return True or a proxy (not False)."""
    specs = load_specs
    result = sparsity_condition_full_beirat_no_total_cap(
        period=20,  # age 50: interior, alive, mother alive, not retired
        lagged_choice=UNEMP_NO_CARE,
        already_retired=0,
        education=0,
        health=GOOD_HEALTH,
        partner_state=0,
        mother_adl=1,
        mother_dead=0,
        care_demand=1,
        job_before_caregiving=0,
        full_leave_year_used=full_leave_year_used,
        job_offer=1,
        caregiving_type=1,
        model_specs=specs,
    )
    assert result is True or isinstance(result, dict)


def _sparsity_proxy(load_specs, **overrides):
    """Helper to call the sparsity function with safe defaults + overrides."""
    base = {
        "period": 20,
        "lagged_choice": UNEMP_NO_CARE,
        "already_retired": 0,
        "education": 0,
        "health": GOOD_HEALTH,
        "partner_state": 0,
        "mother_adl": 0,
        "mother_dead": 0,
        "care_demand": NO_CARE_DEMAND,
        "job_before_caregiving": 0,
        "full_leave_year_used": 0,
        "job_offer": 1,
        "caregiving_type": 1,
        "model_specs": load_specs,
    }
    base.update(overrides)
    return sparsity_condition_full_beirat_no_total_cap(**base)


PROXY_BRANCH_CASES = [
    # (branch_name, overrides that trigger that proxy branch)
    ("dead", {"health": DEAD}),
    ("mother_long_dead", {"mother_dead": PARENT_LONGER_DEAD}),
    (
        "age_past_max_ret_plus_one",
        # period=40 -> age=70 > max_ret_age (67) + 1 = 68; need
        # already_retired=1 to pass the earlier "age > max+1 & already_retired
        # != 1" gate, and lagged_choice retired so we don't hit unemp-after-SRA.
        {"period": 40, "lagged_choice": RETIRED_NO_CARE, "already_retired": 1},
    ),
    (
        "retired_in_past",
        # period=35 -> age=65; lagged retired, age <= max+1; reaches the
        # is_retired-lagged proxy branch.
        {"period": 35, "lagged_choice": RETIRED_NO_CARE, "already_retired": 1},
    ),
    (
        "pre_caregiving_age",
        # period=5 -> age=35 < start_age_caregiving (40).
        {"period": 5, "lagged_choice": UNEMP_NO_CARE},
    ),
]


@pytest.mark.parametrize(
    "branch_name, overrides", PROXY_BRANCH_CASES, ids=[c[0] for c in PROXY_BRANCH_CASES]
)
def test_sparsity_proxy_keys_have_no_total_cap_key(branch_name, overrides, load_specs):
    """Every proxy returned by the new sparsity function must drop the
    ``years_leave_used_total`` key and keep ``full_leave_year_used``."""
    result = _sparsity_proxy(load_specs, **overrides)
    assert isinstance(
        result, dict
    ), f"branch {branch_name} expected a state-proxy dict, got {result!r}"
    assert (
        "years_leave_used_total" not in result
    ), f"branch {branch_name} leaked years_leave_used_total: {sorted(result.keys())}"
    assert (
        "full_leave_year_used" in result
    ), f"branch {branch_name} dropped full_leave_year_used: {sorted(result.keys())}"


# =====================================================================================
# Group 6: Factory wiring
# =====================================================================================


def test_factory_returns_expected_callables():
    """The factory must return exactly the four expected functions."""
    funcs = create_state_space_functions_full_beirat_no_total_cap()

    assert set(funcs.keys()) == {
        "state_specific_choice_set",
        "next_period_deterministic_state",
        "next_period_experience",
        "sparsity_condition",
    }
    assert (
        funcs["state_specific_choice_set"] is state_specific_choice_set_with_caregiving
    )
    assert (
        funcs["next_period_deterministic_state"]
        is next_period_deterministic_state_full_beirat_no_total_cap
    )
    assert (
        funcs["next_period_experience"]
        is get_next_period_experience_caregiving_leave_full_beirat_no_total_cap
    )
    assert funcs["sparsity_condition"] is sparsity_condition_full_beirat_no_total_cap
