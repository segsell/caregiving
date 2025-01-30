"""Shared model specifications and auxiliary functions."""

import jax.numpy as jnp

BASE_YEAR = 2015

N_PERIODS_SIM = 30

MALE = 1
FEMALE = 2

MIN_AGE_SIM = 40
MAX_AGE_SIM = 70
MIN_AGE = 40
MAX_AGE = 70

AGE_40 = 40
AGE_45 = 45
AGE_50 = 50
AGE_55 = 55
AGE_60 = 60
AGE_65 = 65
AGE_70 = 70
AGE_75 = 75
AGE_80 = 80
AGE_85 = 85
AGE_90 = 90
AGE_95 = 95
AGE_100 = 100
AGE_105 = 105

AGE_BINS = [
    (AGE_40 - MIN_AGE, AGE_45 - MIN_AGE),
    (AGE_45 - MIN_AGE, AGE_50 - MIN_AGE),
    (AGE_50 - MIN_AGE, AGE_55 - MIN_AGE),
    (AGE_55 - MIN_AGE, AGE_60 - MIN_AGE),
    (AGE_60 - MIN_AGE, AGE_65 - MIN_AGE),
    (AGE_65 - MIN_AGE, AGE_70 - MIN_AGE),
    (AGE_70 - MIN_AGE, AGE_75 - MIN_AGE),
]

AGE_BINS_SIM = [
    (AGE_40, AGE_45),
    (AGE_45, AGE_50),
    (AGE_50, AGE_55),
    (AGE_55, AGE_60),
    (AGE_60, AGE_65),
    (AGE_65, AGE_70),
]

AGE_BINS_SIM_SHORT = [
    (AGE_50, AGE_55),
    (AGE_55, AGE_60),
    (AGE_60, AGE_65),
    (AGE_65, AGE_70),
]

PARENT_AGE_BINS_SIM = [
    (AGE_65, AGE_70),
    (AGE_70, AGE_75),
    (AGE_75, AGE_80),
    (AGE_80, AGE_85),
    (AGE_85, AGE_90),
    (AGE_90, AGE_100),
]

AGE_BIN_0_TO_3 = 0
AGE_BIN_4_TO_6 = 1
AGE_BIN_7_TO_9 = 2

GOOD_HEALTH = 0
BAD_HEALTH = 1
DEAD = 2
MEDIUM_HEALTH = -99

EARLY_RETIREMENT_AGE = 60
RETIREMENT_AGE = 65


TOTAL_WEEKLY_HOURS = 80
WEEKLY_HOURS_PART_TIME = 20
WEEKLY_HOURS_FULL_TIME = 40
WEEKLY_INTENSIVE_INFORMAL_HOURS = 14  # (21 + 7) / 2

N_WEEKS = 4.33333333333333333
N_MONTHS = 12

PART_TIME_HOURS = 20 * N_WEEKS * N_MONTHS
FULL_TIME_HOURS = 40 * N_WEEKS * N_MONTHS

MAX_LEISURE_HOURS = TOTAL_WEEKLY_HOURS * N_WEEKS * N_MONTHS


# ==============================================================================
# Labor Choices
# ==============================================================================

# ALL = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
# NO_WORK = jnp.array([0, 1, 2, 3])
# PART_TIME = jnp.array([4, 5, 6, 7])
# FULL_TIME = jnp.array([8, 9, 10, 11])
# RETIREMENT = jnp.array([12, 13, 14, 15])
# WORK_NO_CARE = jnp.array([0, 4, 8])
# WORK_AND_NO_WORK = ALL

ALL = jnp.array([0, 1, 2, 3])
RETIREMENT = jnp.array([0])
UNEMPLOYED = jnp.array([1])
PART_TIME = jnp.array([2])
FULL_TIME = jnp.array([3])
WORK_AND_NO_WORK = jnp.array([0, 1, 2, 3])

NOT_WORKING = jnp.concatenate([UNEMPLOYED, RETIREMENT])
WORK = jnp.concatenate([PART_TIME, FULL_TIME])
PART_TIME_AND_NO_WORK = jnp.concatenate([PART_TIME, NOT_WORKING])
FULL_TIME_AND_NO_WORK = jnp.concatenate([FULL_TIME, NOT_WORKING])

NO_RETIREMENT = jnp.concatenate([UNEMPLOYED, PART_TIME, FULL_TIME])

# ==============================================================================
# Caregiving Choices
# ==============================================================================

# NO_CARE = jnp.array([0, 4, 8, 12])
# FORMAL_CARE = jnp.array([1, 5, 9, 13])  # Only nursing home!
# PURE_INFORMAL_CARE = jnp.array([2, 6, 10, 14])
# COMBINATION_CARE = jnp.array([3, 7, 11, 15])

# INFORMAL_CARE = jnp.array(
#     list(set(PURE_INFORMAL_CARE.tolist() + COMBINATION_CARE.tolist())),
# )

# CARE = jnp.concatenate([INFORMAL_CARE, COMBINATION_CARE, FORMAL_CARE])
# CARE_AND_NO_CARE = jnp.concatenate(
#     [NO_CARE, FORMAL_CARE, INFORMAL_CARE, COMBINATION_CARE],
# )
# FORMAL_CARE_AND_NO_CARE = jnp.concatenate([NO_CARE, FORMAL_CARE])
# PURE_INFORMAL_CARE_AND_NO_CARE = jnp.concatenate([NO_CARE, PURE_INFORMAL_CARE])

# # For NO_INFORMAL_CARE and NO_FORMAL_CARE, we need to perform set operations before
# # converting to JAX arrays.
# # This is because JAX doesn't support direct set operations.
# # Convert the results of set operations to lists, then to JAX arrays.
# NO_INFORMAL_CARE = jnp.array(list(set(ALL.tolist()) - set(INFORMAL_CARE.tolist())))
# NO_COMBINATION_CARE = jnp.array(
#     list(set(ALL.tolist()) - set(COMBINATION_CARE.tolist())),
# )
# NO_FORMAL_CARE = jnp.array(list(set(ALL.tolist()) - set(FORMAL_CARE.tolist())))

# ==============================================================================
# Labor conditions
# ==============================================================================


def is_working(choice):
    return jnp.any(choice == WORK)


def is_full_time(choice):
    return jnp.any(choice == FULL_TIME)


def is_part_time(choice):
    return jnp.any(choice == PART_TIME)


def is_unemployed(choice):
    return jnp.any(choice == UNEMPLOYED)


def is_retired(choice):
    return jnp.any(choice == RETIREMENT)


def is_not_working(choice):
    return jnp.any(choice == NOT_WORKING)


# ==============================================================================
# Age of youngest child
# ==============================================================================


def is_child_age_0_to_3(age_youngest_child):
    return age_youngest_child == AGE_BIN_0_TO_3


def is_child_age_4_to_6(age_youngest_child):
    return age_youngest_child == AGE_BIN_4_TO_6


def is_child_age_7_to_9(age_youngest_child):
    return age_youngest_child == AGE_BIN_7_TO_9


# ==============================================================================
# Caregiving
# ==============================================================================


# def is_no_care(lagged_choice):
#     return jnp.any(lagged_choice == NO_CARE)


# def is_informal_care(lagged_choice):
#     # intensive only here
#     return jnp.any(lagged_choice == INFORMAL_CARE)


# def is_pure_informal_care(lagged_choice):
#     # intensive only here
#     return jnp.any(lagged_choice == PURE_INFORMAL_CARE)


# def is_no_informal_care(lagged_choice):
#     # intensive only here
#     return jnp.all(lagged_choice != INFORMAL_CARE)


# def is_formal_care(lagged_choice):
#     return jnp.any(lagged_choice == FORMAL_CARE)


# def is_no_formal_care(lagged_choice):
#     return jnp.all(lagged_choice != FORMAL_CARE)


# def is_combination_care(lagged_choice):
#     return jnp.any(lagged_choice == COMBINATION_CARE)


# ==============================================================================
# Parental health
# ==============================================================================


# def is_good_health(parental_health):
#     return jnp.any(parental_health == GOOD_HEALTH)


# def is_medium_health(parental_health):
#     return jnp.any(parental_health == MEDIUM_HEALTH)


# def is_bad_health(parental_health):
#     return jnp.any(parental_health == BAD_HEALTH)


# ==============================================================================
# Age range
# ==============================================================================


def is_in_age_range(age, age_low, age_high):
    return jnp.logical_and(age >= age_low, age < age_high)
