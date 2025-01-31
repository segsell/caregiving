# import numpy as np

# from caregiving.model.shared import (
#     AGE_50,
#     # BAD_HEALTH,
#     # CARE_AND_NO_CARE,
#     # FORMAL_CARE,
#     # FORMAL_CARE_AND_NO_CARE,
#     FULL_TIME_AND_NO_WORK,
#     INFORMAL_CARE,
#     NO_CARE,
#     NO_RETIREMENT,
#     NOT_WORKING,
#     PART_TIME_AND_NO_WORK,
#     # PURE_INFORMAL_CARE,
#     RETIREMENT,
#     WORK_AND_NO_WORK,
#     # is_formal_care,
#     is_retired,
# )

# from caregiving.model.state_space import update_endog_state


# def create_state_space_functions_no_informal_care():
#     return {
#         "get_next_period_state": update_endog_state,
#         "get_state_specific_choice_set": get_choice_set_no_informal_care,
#     }


# def create_state_space_functions_only_informal_care():
#     return {
#         "get_next_period_state": update_endog_state,
#         "get_state_specific_choice_set": get_choice_set_only_informal_care,
#     }


# ==============================================================================
# State transition
# ==============================================================================


# def get_choice_set_no_informal_care(
#     period,
#     lagged_choice,
#     part_time_offer,
#     full_time_offer,
#     mother_health,
#     options,
# ):
#     """Get feasible choice set for current parent state.

#     # if ((mother_alive == 1) & (mother_health in [MEDIUM_HEALTH, BAD_HEALTH])) | ( #
#     (father_alive == 1) & (father_health in [MEDIUM_HEALTH, BAD_HEALTH]) # ):

#     if experience == options["experience_cap"]:     feasible_choice_set = [i for i in
#     feasible_choice_set if i in NO_WORK]

#     # elif period + options["start_age"] > options["retirement_age"]: #
#     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

#     # elif (age > EARLY_RETIREMENT_AGE) & lagged_choice in NO_WORK: #
#     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

#     """
#     age = options["start_age"] + period

#     _feasible_choice_set_all = np.arange(options["n_choices"])

#     if (mother_health == BAD_HEALTH) & (age >= AGE_50):
#         feasible_choice_set = [
#             i for i in _feasible_choice_set_all if i in FORMAL_CARE_AND_NO_CARE
#         ]

#         # Absorbing nursing home
#         if is_formal_care(lagged_choice):
#             feasible_choice_set = [i for i in feasible_choice_set if i in FORMAL_CARE]
#     else:
#         feasible_choice_set = [i for i in _feasible_choice_set_all if i in NO_CARE]

#     if age < options["min_ret_age"]:
#         feasible_choice_set = [i for i in feasible_choice_set if i in NO_RETIREMENT]

#     if age >= options["max_ret_age"]:
#         feasible_choice_set = RETIREMENT
#     elif is_retired(lagged_choice):
#         feasible_choice_set = [i for i in feasible_choice_set if i in RETIREMENT]
#     elif (full_time_offer == 0) & (part_time_offer == 1):
#         feasible_choice_set = [
#             i for i in feasible_choice_set if i in PART_TIME_AND_NO_WORK
#         ]
#     elif (full_time_offer == 1) & (part_time_offer == 0):
#         feasible_choice_set = [
#             i for i in feasible_choice_set if i in FULL_TIME_AND_NO_WORK
#         ]
#     elif (full_time_offer == 1) & (part_time_offer == 1):
#         feasible_choice_set = [i for i in feasible_choice_set if i in WORK_AND_NO_WORK]
#     else:
#         feasible_choice_set = [i for i in feasible_choice_set if i in NOT_WORKING]

#     return np.array(feasible_choice_set)


# def get_choice_set_only_informal_care(
#     period,
#     lagged_choice,
#     part_time_offer,
#     full_time_offer,
#     mother_health,
#     options,
# ):
#     """Get feasible choice set for current parent state.

#     # if ((mother_alive == 1) & (mother_health in [MEDIUM_HEALTH, BAD_HEALTH])) | ( #
#     (father_alive == 1) & (father_health in [MEDIUM_HEALTH, BAD_HEALTH]) # ):

#     if experience == options["experience_cap"]:     feasible_choice_set = [i for i in
#     feasible_choice_set if i in NO_WORK]

#     # elif period + options["start_age"] > options["retirement_age"]: #
#     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

#     # elif (age > EARLY_RETIREMENT_AGE) & lagged_choice in NO_WORK: #
#     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

#     """
#     age = options["start_age"] + period

#     _feasible_choice_set_all = np.arange(options["n_choices"])

#     # Can only provide care if mother is alive and in bad health
#     if (mother_health == BAD_HEALTH) & (age >= AGE_50):
#         feasible_choice_set = [
#             i for i in _feasible_choice_set_all if i in INFORMAL_CARE
#         ]

#     else:
#         feasible_choice_set = [i for i in _feasible_choice_set_all if i in NO_CARE]

#     if age < options["min_ret_age"]:
#         feasible_choice_set = [i for i in feasible_choice_set if i in NO_RETIREMENT]

#     if age >= options["max_ret_age"]:
#         feasible_choice_set = RETIREMENT
#     elif is_retired(lagged_choice):
#         feasible_choice_set = [i for i in feasible_choice_set if i in RETIREMENT]
#     elif (full_time_offer == 0) & (part_time_offer == 1):
#         feasible_choice_set = [
#             i for i in feasible_choice_set if i in PART_TIME_AND_NO_WORK
#         ]
#     elif (full_time_offer == 1) & (part_time_offer == 0):
#         feasible_choice_set = [
#             i for i in feasible_choice_set if i in FULL_TIME_AND_NO_WORK
#         ]
#     elif (full_time_offer == 1) & (part_time_offer == 1):
#         feasible_choice_set = [i for i in feasible_choice_set if i in WORK_AND_NO_WORK]
#     else:
#         feasible_choice_set = [i for i in feasible_choice_set if i in NOT_WORKING]

#     return np.array(feasible_choice_set)
