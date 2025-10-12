"""Shared constants and helpers for the no-care-demand counterfactual.

Contains the reduced 4-state labor choice arrays and helper predicates.
"""

import jax.numpy as jnp

# ====================================================================================
# Reduced model: Labor Choices (4 discrete choices)
# 0 retirement, 1 unemployed, 2 part-time, 3 full-time
# ====================================================================================

ALL = jnp.arange(4)

RETIREMENT = jnp.array([0])
UNEMPLOYED = jnp.array([1])
PART_TIME = jnp.array([2])
FULL_TIME = jnp.array([3])

WORK = jnp.concatenate([PART_TIME, FULL_TIME])
NOT_WORKING = jnp.concatenate([UNEMPLOYED, RETIREMENT])

# Provide alias arrays with _NO_CARE_DEMAND suffix to match counterfactual usage
ALL_NO_CARE_DEMAND = ALL.copy()
RETIREMENT_NO_CARE_DEMAND = RETIREMENT.copy()
UNEMPLOYED_NO_CARE_DEMAND = UNEMPLOYED.copy()
PART_TIME_NO_CARE_DEMAND = PART_TIME.copy()
FULL_TIME_NO_CARE_DEMAND = FULL_TIME.copy()
WORK_NO_CARE_DEMAND = WORK.copy()
NOT_WORKING_NO_CARE_DEMAND = NOT_WORKING.copy()

WORK_AND_UNEMPLOYED_NO_CARE_DEMAND = jnp.concatenate([UNEMPLOYED, PART_TIME, FULL_TIME])
WORK_AND_RETIREMENT_NO_CARE_DEMAND = jnp.concatenate([RETIREMENT, PART_TIME, FULL_TIME])


# ====================================================================================
# Helper predicates
# ====================================================================================


def is_working(choice):
    """Check if the agent is working."""
    return jnp.any(choice == WORK)


def is_full_time(choice):
    """Check if the agent is working full-time."""
    return jnp.any(choice == FULL_TIME)


def is_part_time(choice):
    """Check if the agent is working part-time."""
    return jnp.any(choice == PART_TIME)


def is_unemployed(choice):
    """Check if the agent is unemployed."""
    return jnp.any(choice == UNEMPLOYED)


def is_retired(choice):
    """Check if the agent is retired."""
    return jnp.any(choice == RETIREMENT)


def is_not_working(choice):
    """Check if the agent is not working."""
    return jnp.any(choice == NOT_WORKING)
