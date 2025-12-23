"""Tests for the state_space module."""

from itertools import product

import jax
import numpy as np
import pytest

from caregiving.model.shared import (
    ALL_NO_CARE,
    NOT_WORKING_NO_CARE,
    RETIREMENT_NO_CARE,
    UNEMPLOYED_NO_CARE,
    WORK_AND_UNEMPLOYED_NO_CARE,
)
from caregiving.model.state_space import state_specific_choice_set

jax.config.update("jax_enable_x64", True)

# tests of choice set
PERIOD_GRID = np.linspace(10, 30, 3)
CHOICE_SET = WORK_AND_UNEMPLOYED_NO_CARE
LAGGED_CHOICE_SET_WORKING_LIFE = WORK_AND_UNEMPLOYED_NO_CARE
JOB_OFFER_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, lagged_choice, job_offer",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET_WORKING_LIFE, JOB_OFFER_GRID)),
)
def test_choice_set_under_63(period, lagged_choice, job_offer):
    """Test choice set for ages under 63."""
    model_specs = {
        "start_age": 25,
        "min_SRA": 65,
        "min_ret_age": 63,
        "ret_years_before_SRA": 4,
    }
    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        job_offer=job_offer,
        health=1,
        model_specs=model_specs,
    )
    if job_offer == 1:
        # if sex == 0:
        #     assert np.all(choice_set == np.array([1, 3]))
        # else:
        assert np.all(choice_set == WORK_AND_UNEMPLOYED_NO_CARE)
    else:
        assert np.all(choice_set == UNEMPLOYED_NO_CARE)


PERIOD_GRID = np.linspace(25, 45, 2)
FULL_CHOICE_SET = ALL_NO_CARE
JOB_OFFER_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, lagged_choice, job_offer",
    list(product(PERIOD_GRID, FULL_CHOICE_SET, JOB_OFFER_GRID)),
)
def test_choice_set_over_63_under_72(period, lagged_choice, job_offer):
    """Test choice set for ages over 63 and under 72."""
    model_specs = {
        "start_age": 30,
        "min_SRA": 67,
        "min_ret_age": 63,
        "max_ret_age": 72,
        "ret_years_before_SRA": 4,
    }

    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        job_offer=job_offer,
        health=1,
        model_specs=model_specs,
    )
    age = period + model_specs["start_age"]
    min_ret_age = model_specs["min_ret_age"]

    if lagged_choice == RETIREMENT_NO_CARE:
        assert np.all(choice_set == RETIREMENT_NO_CARE)
    else:
        if age < min_ret_age:
            # Not old enough to retire. Check if job is offered
            if job_offer == 1:
                # if sex == 0:
                #     assert np.all(choice_set == np.array([1, 3]))
                # else:
                assert np.all(choice_set == WORK_AND_UNEMPLOYED_NO_CARE)
            else:
                assert np.all(choice_set == UNEMPLOYED_NO_CARE)
        else:
            if age < model_specs["max_ret_age"]:
                if job_offer == 1:
                    # if sex == 0:
                    #     assert np.all(choice_set == np.array([0, 1, 3]))
                    # else:
                    assert np.all(choice_set == ALL_NO_CARE)
                else:
                    assert np.all(choice_set == NOT_WORKING_NO_CARE)
            else:
                assert np.all(choice_set == RETIREMENT_NO_CARE)


PERIOD_GRID = np.linspace(47, 55, 1)
LAGGED_CHOICE_SET = ALL_NO_CARE


@pytest.mark.parametrize(
    "period, lagged_choice",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET)),
)
def test_choice_set_over_72(period, lagged_choice):
    """Test choice set for ages over 72."""
    model_specs = {
        "start_age": 25,
        "min_SRA": 65,
        "min_ret_age": 63,
        "max_ret_age": 72,
        "ret_years_before_SRA": 4,
    }

    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        job_offer=0,
        health=1,
        model_specs=model_specs,
    )
    assert np.all(choice_set == RETIREMENT_NO_CARE)
