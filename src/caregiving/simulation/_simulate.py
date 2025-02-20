# """Simulate moments for estimation."""

# import jax
# import jax.numpy as jnp
# import numpy as np
# import pandas as pd
# import pytask
# from sklearn.linear_model import LogisticRegression

# from caregiving.model.budget import calc_net_income_pensions
# from caregiving.model.shared import (
#     AGE_BINS_SIM,
#     AGE_BINS_SIM_SHORT,
#     BAD_HEALTH,
#     BETA,
#     COMBINATION_CARE,
#     DEAD,
#     FORMAL_CARE,
#     FULL_TIME,
#     INFORMAL_CARE,
#     MAX_AGE_SIM,
#     MIN_AGE_SIM,
#     NO_CARE,
#     NO_COMBINATION_CARE,
#     NO_INFORMAL_CARE,
#     NOT_WORKING,
#     PART_TIME,
#     PURE_INFORMAL_CARE,
#     RETIREMENT,
#     RETIREMENT_AGE,
# )


# def simulate_moments(arr, idx):
#     """Df has multiindex ["period", "agent"] necessary?

#     column_indices = {col: idx for idx, col in enumerate(sim.columns)} idx =
#     column_indices.copy() arr = jnp.asarray(sim)

#      # share_informal_care_by_age_bin = get_share_by_type_by_age_bin( #     arr, #
#     ind=idx, #     care_type=ALL, #     lagged_choice=INFORMAL_CARE, # )


#     #
#     share_not_working_no_informal_care = get_share_by_type(
#         arr,
#         ind=idx,
#         choice=NO_WORK,
#         care_type=NO_INFORMAL_CARE,
#     )
#     share_part_time_no_informal_care = get_share_by_type(
#         arr,
#         ind=idx,
#         choice=PART_TIME,
#         care_type=NO_INFORMAL_CARE,
#     )
#     share_full_time_no_informal_care = get_share_by_type(
#         arr,
#         ind=idx,
#         choice=FULL_TIME,
#         care_type=NO_INFORMAL_CARE,
#     )

#     share_not_working_informal_care = get_share_by_type(
#         arr,
#         ind=idx,
#         choice=NO_WORK,
#         care_type=INFORMAL_CARE,
#     )
#     share_part_time_informal_care = get_share_by_type(
#         arr,
#         ind=idx,
#         choice=PART_TIME,
#         care_type=INFORMAL_CARE,
#     )
#     share_full_time_informal_care = get_share_by_type(
#         arr,
#         ind=idx,
#         choice=FULL_TIME,
#         care_type=INFORMAL_CARE,
#     )


#     [share_not_working_no_informal_care]
#         + [share_part_time_no_informal_care]
#         + [share_full_time_no_informal_care]
#         + [share_not_working_informal_care]
#         + [share_part_time_informal_care]
#         + [share_full_time_informal_care]
#         +


#     + care_mix_informal_by_mother_age_bin
#         # + care_mix_formal_by_mother_age_bin
#         # + care_mix_combination_by_mother_age_bi

#     share_not_working_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=NO_WORK,
#     )
#     share_working_part_time_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=PART_TIME,
#     )
#     share_working_full_time_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=FULL_TIME,
#     )

#     """
#     # ================================================================================
#     # Employment by age
#     # ================================================================================

#     share_out_of_labor_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=NOT_WORKING,
#     )
#     share_working_part_time_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=PART_TIME,
#     )
#     share_working_full_time_by_age = get_share_by_age(
#         arr,
#         ind=idx,
#         choice=FULL_TIME,
#     )

#     # ================================================================================
#     # Savings rate
#     # ================================================================================

#     savings_rate_coeffs = fit_ols(
#         x=arr[
#             :,
#             [
#                 idx["age"],
#                 idx["age_squared"],
#                 # idx["high_educ"],
#                 idx["choice_part_time"],
#                 idx["choice_full_time"],
#             ],
#         ],
#         y=arr[:, idx["savings_rate"]],
#     )

#     # ================================================================================
#     # Care mix
#     # ================================================================================

#     no_care_coeffs = fit_logit(
#         x=arr[
#             :,
#             [
#                 idx["mother_age_bin_70_75"],
#                 idx["mother_age_bin_75_80"],
#                 idx["mother_age_bin_80_85"],
#                 idx["mother_age_bin_85_plus"],
#             ],
#         ],
#         y=arr[:, idx["choice_no_care"]],
#     )
#     pure_informal_care_coeffs = fit_logit(
#         x=arr[
#             :,
#             [
#                 idx["mother_age_bin_70_75"],
#                 idx["mother_age_bin_75_80"],
#                 idx["mother_age_bin_80_85"],
#                 idx["mother_age_bin_85_plus"],
#             ],
#         ],
#         y=arr[:, idx["choice_pure_informal_care"]],
#     )
#     combination_care_coeffs = fit_logit(
#         x=arr[
#             :,
#             [
#                 idx["mother_age_bin_70_75"],
#                 idx["mother_age_bin_75_80"],
#                 idx["mother_age_bin_80_85"],
#                 idx["mother_age_bin_85_plus"],
#             ],
#         ],
#         y=arr[:, idx["choice_combination_care"]],
#     )
#     formal_care_coeffs = fit_logit(
#         x=arr[
#             :,
#             [
#                 idx["mother_age_bin_70_75"],
#                 idx["mother_age_bin_75_80"],
#                 idx["mother_age_bin_80_85"],
#                 idx["mother_age_bin_85_plus"],
#             ],
#         ],
#         y=arr[:, idx["choice_formal_care"]],
#     )

#     # ================================================================================
#     # Labor shares by informal caregiving status
#     # ================================================================================

#     # share_not_working_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#     #     arr,
#     #     ind=idx,
#     #     choice=NO_WORK,
#     #     care_type=NO_INFORMAL_CARE,
#     #     age_bins=AGE_BINS_SIM,
#     # )
#     # share_part_time_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#     #     arr,
#     #     ind=idx,
#     #     choice=PART_TIME,
#     #     care_type=INFORMAL_CARE,
#     #     age_bins=AGE_BINS_SIM,
#     # )
#     # share_full_time_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#     #     arr,
#     #     ind=idx,
#     #     choice=FULL_TIME,
#     #     care_type=NO_INFORMAL_CARE,
#     #     age_bins=AGE_BINS_SIM,
#     # )

#     share_not_working_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#         arr,
#         ind=idx,
#         choice=NOT_WORKING,
#         care_type=INFORMAL_CARE,
#         age_bins=AGE_BINS_SIM,
#     )
#     share_part_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#         arr,
#         ind=idx,
#         choice=PART_TIME,
#         care_type=INFORMAL_CARE,
#         age_bins=AGE_BINS_SIM,
#     )
#     share_full_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#         arr,
#         ind=idx,
#         choice=FULL_TIME,
#         care_type=INFORMAL_CARE,
#         age_bins=AGE_BINS_SIM,
#     )

#     # ================================================================================
#     # Share caregiving by age bin
#     # ================================================================================

#     # share_informal_care_by_age_bin = get_share_by_type_by_age_bin(
#     #     arr,
#     #     ind=idx,
#     #     choice=INFORMAL_CARE,
#     #     care_type=ALL,
#     #     age_bins=AGE_BINS_SIM,
#     # )

#     # ================================================================================
#     # Share care by mother's health
#     # ================================================================================

#     # no_care_mother_health = get_share_care_by_parental_health(
#     #     arr,
#     #     ind=idx,
#     #     care_choice=NO_CARE,
#     #     parent="mother",
#     # )
#     # informal_care_mother_health = get_share_care_by_parental_health(
#     #     arr,
#     #     ind=idx,
#     #     care_choice=PURE_INFORMAL_CARE,
#     #     parent="mother",
#     # )
#     # formal_care_mother_health = get_share_care_by_parental_health(
#     #     arr,
#     #     ind=idx,
#     #     care_choice=PURE_FORMAL_CARE,
#     #     parent="mother",
#     # )
#     # combination_care_mother_health = get_share_care_by_parental_health(
#     #     arr,
#     #     ind=idx,
#     #     care_choice=COMBINATION_CARE,
#     #     parent="mother",
#     # )

#     # no_care_mother_health_has_sibling = (
#     #     get_share_care_by_parental_health_and_presence_of_sibling(
#     #         arr,
#     #         ind=idx,
#     #         care_choice=NO_CARE,
#     #         has_sibling=True,
#     #         parent="mother",
#     #     )
#     # )
#     # informal_care_mother_health_has_sibling = (
#     #     get_share_care_by_parental_health_and_presence_of_sibling(
#     #         arr,
#     #         ind=idx,
#     #         care_choice=PURE_INFORMAL_CARE,
#     #         has_sibling=True,
#     #         parent="mother",
#     #     )
#     # )
#     # formal_care_mother_health_has_sibling = (
#     #     get_share_care_by_parental_health_and_presence_of_sibling(
#     #         arr,
#     #         ind=idx,
#     #         care_choice=PURE_FORMAL_CARE,
#     #         has_sibling=True,
#     #         parent="mother",
#     #     )
#     # )
#     # combination_care_mother_health_has_sibling = (
#     #     get_share_care_by_parental_health_and_presence_of_sibling(
#     #         arr,
#     #         ind=idx,
#     #         care_choice=COMBINATION_CARE,
#     #         has_sibling=True,
#     #         parent="mother",
#     #     )
#     # )

#     # ================================================================================
#     # Employment transitions
#     # ================================================================================

#     no_work_to_no_work = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NOT_WORKING,
#         current_choice=NOT_WORKING,
#     )
#     no_work_to_part_time = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NOT_WORKING,
#         current_choice=PART_TIME,
#     )
#     no_work_to_full_time = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NOT_WORKING,
#         current_choice=FULL_TIME,
#     )

#     part_time_to_no_work = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=PART_TIME,
#         current_choice=NOT_WORKING,
#     )
#     part_time_to_part_time = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=PART_TIME,
#         current_choice=PART_TIME,
#     )
#     part_time_to_full_time = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=PART_TIME,
#         current_choice=FULL_TIME,
#     )

#     full_time_to_no_work = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=FULL_TIME,
#         current_choice=NOT_WORKING,
#     )
#     full_time_to_part_time = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=FULL_TIME,
#         current_choice=PART_TIME,
#     )
#     full_time_to_full_time = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=FULL_TIME,
#         current_choice=FULL_TIME,
#     )

#     # ================================================================================
#     # Caregiving transitions
#     # ================================================================================

#     no_informal_care_to_no_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_INFORMAL_CARE,
#         current_choice=NO_INFORMAL_CARE,
#     )
#     no_informal_care_to_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_INFORMAL_CARE,
#         current_choice=INFORMAL_CARE,
#     )

#     informal_care_to_no_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=INFORMAL_CARE,
#         current_choice=NO_INFORMAL_CARE,
#     )
#     informal_care_to_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=INFORMAL_CARE,
#         current_choice=INFORMAL_CARE,
#     )

#     no_informal_care_to_no_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_INFORMAL_CARE,
#         current_choice=NO_COMBINATION_CARE,
#     )
#     no_informal_care_to_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_INFORMAL_CARE,
#         current_choice=COMBINATION_CARE,
#     )

#     informal_care_to_no_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=INFORMAL_CARE,
#         current_choice=NO_COMBINATION_CARE,
#     )
#     informal_care_to_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=INFORMAL_CARE,
#         current_choice=COMBINATION_CARE,
#     )

#     no_formal_care_to_no_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_COMBINATION_CARE,
#         current_choice=NO_INFORMAL_CARE,
#     )
#     no_formal_care_to_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_COMBINATION_CARE,
#         current_choice=INFORMAL_CARE,
#     )

#     formal_care_to_no_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=COMBINATION_CARE,
#         current_choice=NO_INFORMAL_CARE,
#     )
#     formal_care_to_informal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=COMBINATION_CARE,
#         current_choice=INFORMAL_CARE,
#     )

#     no_formal_care_to_no_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_COMBINATION_CARE,
#         current_choice=NO_COMBINATION_CARE,
#     )
#     no_formal_care_to_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=NO_COMBINATION_CARE,
#         current_choice=COMBINATION_CARE,
#     )

#     formal_care_to_no_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=COMBINATION_CARE,
#         current_choice=NO_COMBINATION_CARE,
#     )
#     formal_care_to_formal_care = get_transition(
#         arr,
#         ind=idx,
#         lagged_choice=COMBINATION_CARE,
#         current_choice=COMBINATION_CARE,
#     )

#     # ================================================================================
#     # Moments matrix
#     # ================================================================================

#     return jnp.asarray(
#         # employment shares
#         share_out_of_labor_by_age
#         + share_working_part_time_by_age
#         + share_working_full_time_by_age
#         # regressions
#         + savings_rate_coeffs.tolist()
#         + no_care_coeffs
#         + pure_informal_care_coeffs
#         + combination_care_coeffs
#         + formal_care_coeffs
#         #
#         # employment shares by caregiving status
#         # no informal care
#         # + share_not_working_no_informal_care_by_age_bin
#         # + share_part_time_no_informal_care_by_age_bin
#         # + share_full_time_no_informal_care_by_age_bin
#         # # informal care
#         + share_not_working_informal_care_by_age_bin
#         + share_part_time_informal_care_by_age_bin
#         + share_full_time_informal_care_by_age_bin
#         # #
#         # # share of informal care in total population by age bin
#         # # + share_informal_care_by_age_bin
#         # #
#         # Employment transitions
#         + no_work_to_no_work
#         + no_work_to_part_time
#         + no_work_to_full_time
#         + part_time_to_no_work
#         + part_time_to_part_time
#         + part_time_to_full_time
#         + full_time_to_no_work
#         + full_time_to_part_time
#         + full_time_to_full_time
#         +
#         # # Caregiving transitions
#         no_informal_care_to_no_informal_care
#         + no_informal_care_to_informal_care
#         + informal_care_to_no_informal_care
#         + informal_care_to_informal_care
#         + no_informal_care_to_no_formal_care
#         + no_informal_care_to_formal_care
#         + informal_care_to_no_formal_care
#         + informal_care_to_formal_care
#         + no_formal_care_to_no_informal_care
#         + no_formal_care_to_informal_care
#         + formal_care_to_no_informal_care
#         + formal_care_to_informal_care
#         + no_formal_care_to_no_formal_care
#         + no_formal_care_to_formal_care
#         + formal_care_to_no_formal_care
#         + formal_care_to_formal_care,
#     )


# def fit_ols(x, y):
#     """Fit a linear model using least-squares.

#     coef = coef.T


#     Args:
#         x (jnp.ndarray): Array of shape (n, p) of x-values.
#         y (jnp.ndarray): Array of shape (n, k) of y-values.

#     Returns:
#         coef (np.ndarray): Array of shape (p, k) of coefficients.

#     """
#     intercept = jnp.ones((len(x), 1))
#     features = jnp.concatenate((intercept, jnp.atleast_2d(x)), axis=1)

#     coef, *_ = jnp.linalg.lstsq(features, y, rcond=None)

#     return coef


# def fit_logit(x, y):
#     """Fit a logistic regression model.

#     Args:
#         x (jnp.ndarray): Array of shape (n, p) of x-values.
#         y (jnp.ndarray): Array of shape (n, k) of y-values.

#     Returns:
#         coef (list)): list of length (k + 1,) of coefficients.

#     """
#     # model = LogisticRegression()
#     # model.fit(x, y)

#     # # return jnp.array([model.intercept_[0]] + list(model.coef_[0]))
#     # out = [model.intercept_[0]] + list(model.coef_[0])

#     model = LogisticRegression()
#     try:
#         model.fit(x, y)
#         out = [model.intercept_[0], *model.coef_[0]]
#     except ValueError:
#         out = [np.nan] * (x.shape[1] + 1)

#     return out


# def get_share_by_age(df_arr, ind, choice):
#     """Get share of agents choosing lagged choice by age bin."""
#     choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
#     shares = []

#     for age in range(MIN_AGE_SIM, MAX_AGE_SIM):
#         age_mask = df_arr[:, ind["age"]] == age

#         share = jnp.sum(age_mask & choice_mask) / jnp.sum(age_mask)
#         # period count is always larger than 0! otherwise error
#         shares.append(share)

#     return shares


# def get_share_by_type(df_arr, ind, choice, care_type):
#     """Get share of agents of given care type choosing lagged choice by age bin."""
#     choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
#     care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

#     return jnp.sum(choice_mask & care_type_mask) / jnp.sum(care_type_mask)


# def get_transition(df_arr, ind, lagged_choice, current_choice):
#     """Get transition probability from lagged choice to current choice."""
#     return [
#         jnp.sum(
#             jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
#             & jnp.isin(df_arr[:, ind["choice"]], current_choice),
#         )
#         / jnp.sum(jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)),
#     ]


# def get_share_by_type_by_age_bin(df_arr, ind, choice, care_type, age_bins):
#     """Get share of agents of given care type choosing lagged choice by age bin."""
#     choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
#     type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

#     shares = []
#     for age_bin in age_bins:
#         age_bin_mask = (df_arr[:, ind["age"]] >= age_bin[0]) & (
#             df_arr[:, ind["age"]] < age_bin[1]
#         )
#         share = jnp.sum(choice_mask & type_mask & age_bin_mask) / jnp.sum(
#             type_mask & age_bin_mask,
#         )
#         shares.append(share)

#     return shares


# def get_savings_rate_by_age_bin(arr, ind, care_type):
#     """Get savings rate of given care type by age bin."""
#     care_type_mask = jnp.isin(arr[:, ind["choice"]], care_type)
#     means = []

#     for age_bin in AGE_BINS_SIM:
#         age_bin_mask = (arr[:, ind["age"]] >= age_bin[0]) & (
#             arr[:, ind["age"]] < age_bin[1]
#         )
#         selected_values = jnp.take(
#             arr[:, ind["savings_rate"]],
#             jnp.nonzero(care_type_mask & age_bin_mask)[0],
#         )
#         mean = jnp.sum(selected_values) / jnp.sum(care_type_mask & age_bin_mask)

#         means.append(mean)

#     return means


# def get_share_care_by_parental_health_old(
#     df_arr,
#     ind,
#     care_choice,
#     parent="mother",
# ):
#     """Get share of agents choosing given care choice by parental health."""
#     return [
#         jnp.mean(
#             jnp.isin(df_arr[:, ind["choice"]], care_choice)
#             & (df_arr[:, ind[f"{parent}_health"]], health),
#         )
#         for health in (BAD_HEALTH)
#     ]


# def get_share_care_by_parental_health_and_presence_of_sibling_old(
#     df_arr,
#     ind,
#     care_choice,
#     has_sibling,
#     parent="mother",
# ):
#     """Get share of agents choosing given care choice by parental health."""
#     return [
#         jnp.mean(
#             jnp.isin(df_arr[:, ind["choice"]], care_choice)
#             & (df_arr[:, ind["has_sibling"]] == has_sibling)
#             & (df_arr[:, ind[f"{parent}_health"]], health),
#         )
#         for health in (BAD_HEALTH)
#     ]


# def get_share_care_by_parental_health_and_presence_of_sibling(
#     df_arr,
#     ind,
#     care_choice,
#     has_sibling,
#     parent="mother",
# ):
#     """Get share of agents choosing given care choice by parental health."""
#     care_choice_mask = jnp.isin(df_arr[:, ind["choice"]], care_choice)
#     sibling_mask = df_arr[:, ind["has_sibling"]] == has_sibling

#     shares = []
#     for health in [BAD_HEALTH]:
#         health_mask = df_arr[:, ind[f"{parent}_health"]] == health

#         share = jnp.sum(care_choice_mask & sibling_mask & health_mask) / jnp.sum(
#             sibling_mask & health_mask,
#         )
#         shares.append(share)

#     return shares


# def get_share_care_by_parental_health(
#     df_arr,
#     ind,
#     care_choice,
#     parent="mother",
# ):
#     """Get share of agents choosing given care choice by parental health."""
#     care_choice_mask = jnp.isin(df_arr[:, ind["choice"]], care_choice)

#     shares = []
#     for health in [BAD_HEALTH]:
#         health_mask = df_arr[:, ind[f"{parent}_health"]] == health

#         share = jnp.sum(care_choice_mask & health_mask) / jnp.sum(health_mask)
#         shares.append(share)

#     return shares


# def get_care_mix_by_mother_age_bin(df_arr, ind, choice, care_type, age_bins):
#     """Get share of agents of given care type choosing lagged choice by age bin."""
#     choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
#     care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

#     shares = []
#     for age_bin in age_bins:
#         age_bin_mask = (df_arr[:, ind["mother_age"]] >= age_bin[0]) & (
#             df_arr[:, ind["mother_age"]] < age_bin[1]
#         )
#         share = jnp.sum(choice_mask & care_type_mask & age_bin_mask) / jnp.sum(
#             care_type_mask & age_bin_mask,
#         )
#         shares.append(share)

#     return shares


# # ==============================================================================
# # Prepare simulation array
# # ==============================================================================


# # Assuming _assign_working_hours is adapted for JAX, using vmap for batch operations
# # if necessary.
# # Since JAX does not support functions like .apply() directly,
# # you'll likely need to vectorize this logic.


# def create_simulation_array_from_df(data, options, params):
#     """Create simulation array from dict."""
#     data = data.copy()  # Make a copy to avoid modifying a slice

#     options = options["model_params"]
#     n_agents = options["n_agents"]
#     n_periods = options["n_periods"]

#     # Assigning the 'agent' and age-related calculations
#     data.loc[:, "agent"] = jnp.tile(jnp.arange(n_agents), n_periods)
#     period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()

#     data.loc[:, "age"] = options["start_age"] + period_indices
#     data.loc[:, "age_squared"] = data["age"] ** 2

#     # Mother age bins
#     data.loc[:, "mother_alive"] = data["mother_health"] < DEAD

#     data.loc[:, "mother_age"] = options["mother_start_age"] + period_indices
#     data.loc[data["mother_alive"] == False, "mother_age"] = np.nan

#     # Define age bins
#     bins = [65, 70, 75, 80, 85, np.inf]
#     labels = ["65_70", "70_75", "75_80", "80_85", "85_plus"]

#     data["mother_age_bin"] = pd.cut(
#         data["mother_age"],
#         bins=bins,
#         labels=labels,
#         right=False,
#     )

#     dummies = pd.get_dummies(data["mother_age_bin"], prefix="mother_age_bin")
#     data = pd.concat([data, dummies], axis=1)
#     data = data.drop(columns=["mother_age_bin"])

#     # Financial calculations
#     data.loc[:, "wealth"] = data["savings"] + data["consumption"]
#     data.loc[:, "savings_rate"] = jnp.where(
#         jnp.array(data["wealth"]) > 0,
#         jnp.divide(jnp.array(data["savings"]), jnp.array(data["wealth"])),
#         0,
#     )

#     # Squared experience
#     data.loc[:, "experience"] = data["experience"] / 2
#     data.loc[:, "experience_squared"] = data["experience"] ** 2

#     # Employment status
#     data.loc[:, "lagged_part_time"] = jnp.isin(
#         jnp.array(data["lagged_choice"]),
#         PART_TIME,
#     )

#     data.loc[:, "choice_retired"] = jnp.isin(
#         jnp.array(data["choice"]),
#         RETIREMENT,
#     )
#     data.loc[:, "choice_part_time"] = jnp.isin(
#         jnp.array(data["choice"]),
#         PART_TIME,
#     )
#     data.loc[:, "choice_full_time"] = jnp.isin(
#         jnp.array(data["choice"]),
#         FULL_TIME,
#     )
#     # data.loc[:, "choice_informal_care"] = jnp.isin(
#     #     jnp.array(data["choice"]),
#     #     INFORMAL_CARE,
#     # )

#     data.loc[:, "choice_no_care"] = jnp.isin(
#         jnp.array(data["choice"]),
#         NO_CARE,
#     )
#     data.loc[:, "choice_pure_informal_care"] = jnp.isin(
#         jnp.array(data["choice"]),
#         PURE_INFORMAL_CARE,
#     )
#     data.loc[:, "choice_combination_care"] = jnp.isin(
#         jnp.array(data["choice"]),
#         COMBINATION_CARE,
#     )
#     data.loc[:, "choice_formal_care"] = jnp.isin(
#         jnp.array(data["choice"]),
#         FORMAL_CARE,
#     )

#     # Logit regressison: Trim subsample
#     # Define the columns to be set to NaN
#     columns_to_nan = [
#         "choice_no_care",
#         "choice_pure_informal_care",
#         "choice_combination_care",
#         "choice_formal_care",
#     ]
#     columns_to_nan += [col for col in data.columns if col.startswith("mother_age_bin")]

#     # Set the specified columns to NaN where mother_health != 1
#     data.loc[data["mother_health"] != BAD_HEALTH, columns_to_nan] = np.nan

#     # Wage calculations
#     data.loc[:, "log_wage"] = (
#         params["wage_constant"]
#         + params["wage_experience"] * data["experience"]
#         + params["wage_experience_squared"] * data["experience_squared"]
#         # + params["wage_high_education"] * data["high_educ"]
#         + params["wage_part_time"] * data["lagged_part_time"]
#     )

#     data.loc[:, "wage"] = jnp.exp(
#         jnp.array(data["log_wage"]) + jnp.array(data["income_shock"]),
#     )

#     # Working hours and income calculation
#     data.loc[:, "working_hours"] = jax.vmap(_assign_working_hours_vectorized)(
#         data["lagged_choice"].values,
#     )

#     data.loc[:, "income"] = data["working_hours"] * data["wage"]

#     # Create a mapping of column indices
#     column_indices = {col: idx for idx, col in enumerate(data.columns)}

#     data = data.dropna()

#     # return np.asarray(data).astype(float), column_indices
#     return jnp.array(data, dtype=float), column_indices


# @pytask.mark.skip()
# def create_simulation_array_from_df_counterfactual(data, options, params):
#     """Create simulation array from dict."""
#     data = data.copy()  # Make a copy to avoid modifying a slice

#     options = options["model_params"]
#     n_agents = options["n_agents"]
#     n_periods = options["n_periods"]

#     # Assigning the 'agent' and age-related calculations
#     data.loc[:, "agent"] = jnp.tile(jnp.arange(n_agents), n_periods)
#     period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()

#     data.loc[:, "age"] = options["start_age"] + period_indices
#     data.loc[:, "age_squared"] = data["age"] ** 2
#     data.loc[:, "mother_age"] = options["mother_start_age"] + period_indices

#     # Financial calculations
#     data.loc[:, "wealth"] = data["savings"] + data["consumption"]
#     data.loc[:, "savings_rate"] = jnp.where(
#         jnp.array(data["wealth"]) > 0,
#         jnp.divide(jnp.array(data["savings"]), jnp.array(data["wealth"])),
#         0,
#     )

#     data.loc[:, "experience"] = data["experience"] / 2
#     data.loc[:, "experience_squared"] = data["experience"] ** 2

#     # Employment status
#     data.loc[:, "lagged_part_time"] = jnp.isin(
#         jnp.array(data["lagged_choice"]),
#         PART_TIME,
#     ).astype(np.uint8)

#     data.loc[:, "choice_retired"] = jnp.isin(
#         jnp.array(data["choice"]),
#         RETIREMENT,
#     ).astype(np.uint8)
#     data.loc[:, "choice_part_time"] = jnp.isin(
#         jnp.array(data["choice"]),
#         PART_TIME,
#     ).astype(np.uint8)
#     data.loc[:, "choice_full_time"] = jnp.isin(
#         jnp.array(data["choice"]),
#         FULL_TIME,
#     ).astype(np.uint8)
#     # data.loc[:, "choice_informal_care"] = jnp.isin(
#     #     jnp.array(data["choice"]),
#     #     INFORMAL_CARE,
#     # ).astype(np.uint8)

#     # Wage calculations
#     data.loc[:, "log_wage"] = (
#         params["wage_constant"]
#         + params["wage_experience"] * data["experience"]
#         + params["wage_experience_squared"] * data["experience_squared"]
#         # + params["wage_high_education"] * data["high_educ"]
#         + params["wage_part_time"] * data["lagged_part_time"]
#     )

#     data.loc[:, "wage"] = jnp.exp(
#         jnp.array(data["log_wage"]) + jnp.array(data["income_shock"]),
#     )

#     # Working hours and income calculation
#     data.loc[:, "working_hours"] = jax.vmap(_assign_working_hours_vectorized)(
#         data["lagged_choice"].values,
#     )
#     data.loc[:, "labor_income"] = data["wage"] * data["working_hours"]

#     # Unemployment benefits
#     data.loc[:, "means_test"] = (
#         data.loc[:, "savings"] < options["unemployment_wealth_thresh"]
#     )
#     data.loc[:, "unemployment_benefits"] = (
#         data.loc[:, "means_test"] * options["unemployment_benefits"] * 12
#     )

#     # retirement benefits
#     data.loc[:, "pension_factor"] = (
#         1 - (data.loc[:, "age"] - RETIREMENT_AGE) * options["early_retirement_penalty"]
#     )
#     data.loc[:, "retirement_income_gross_one_year"] = (
#         options["pension_point_value"]
#         * data.loc[:, "experience"]
#         * data.loc[:, "pension_factor"]
#         * data.loc[:, "choice_retired"]  # only receive benefits if actually retired
#         * 12
#     )
#     data.loc[:, "retirement_income"] = jax.vmap(calc_net_income_pensions)(
#         data.loc[:, "retirement_income_gross_one_year"].values,
#     )

#     # Cumulative life time income
#     data["cum_labor_income"] = data.groupby(level="agent")["labor_income"].transform(
#         "cumsum",
#     )
#     data["cum_unemployment_benefits"] = data.groupby(level="agent")[
#         "unemployment_benefits"
#     ].transform("cumsum")
#     data["cum_retirement_income"] = data.groupby(level="agent")[
#         "retirement_income"
#     ].transform("cumsum")

#     # Discount factor
#     data["beta"] = BETA ** data.index.get_level_values("period")

#     # Net Present Value for each income stream
#     data["NPV_labor_income"] = data["labor_income"] * data["beta"]
#     data["NPV_unemployment_benefits"] = data["unemployment_benefits"] * data["beta"]
#     data["NPV_retirement_income"] = data["retirement_income"] * data["beta"]

#     # npv_data = data.groupby(level="agent")[
#     #     ["NPV_labor_income", "NPV_unemployment_benefits", "NPV_retirement_income"]
#     # ].sum()

#     # Create a mapping of column indices
#     column_indices = {col: idx for idx, col in enumerate(data.columns)}

#     data = data.dropna()

#     return jnp.array(data), column_indices


# def create_simulation_array(sim_dict, options, params):
#     """Create simulation array from dict."""
#     options = options["model_params"]
#     n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

#     agent = jnp.tile(jnp.arange(n_agents), n_periods)
#     period = sim_dict["period"].ravel()
#     savings = sim_dict["savings"].ravel()
#     consumption = sim_dict["consumption"].ravel()
#     lagged_choice = sim_dict["lagged_choice"].ravel()
#     choice = sim_dict["choice"].ravel()
#     # high_educ = sim_dict["high_educ"].ravel()
#     experience = sim_dict["experience"].ravel()
#     income_shock = sim_dict["income_shock"].ravel()
#     mother_health = sim_dict["mother_health"].ravel()
#     has_sibling = sim_dict["has_sibling"].ravel()

#     wealth = savings + consumption
#     savings_rate = jnp.where(wealth > 0, jnp.divide(savings, wealth), 0)
#     period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()

#     age = options["start_age"] + period_indices
#     age_squared = age**2
#     mother_age = options["mother_start_age"] + period_indices

#     experience_squared = experience**2

#     # Adjusting the logic for PART_TIME and FULL_TIME checks
#     lagged_part_time = jnp.isin(lagged_choice, PART_TIME)

#     is_part_time = jnp.isin(choice, PART_TIME)
#     is_full_time = jnp.isin(choice, FULL_TIME)
#     is_informal_care = jnp.isin(choice, INFORMAL_CARE)

#     log_wage = (
#         params["wage_constant"]
#         + params["wage_experience"] * experience
#         + params["wage_experience_squared"] * experience_squared
#         # + params["wage_high_education"] * high_educ
#         + params["wage_part_time"] * lagged_part_time
#     )

#     wage = jnp.exp(log_wage + income_shock)

#     # Adapt _assign_working_hours to work with vectorized operations in JAX
#     # Example stub for _assign_working_hours function
#     # Vectorize the _assign_working_hours function if it's not already suitable
#     # for vector operations
#     working_hours = jax.vmap(_assign_working_hours_vectorized)(lagged_choice)

#     income = working_hours * wage

#     arr = jnp.column_stack(
#         (
#             agent,
#             period,
#             lagged_choice,
#             choice,
#             age,
#             age_squared,
#             mother_age,
#             wealth,
#             savings_rate,
#             wage,
#             working_hours,
#             income,
#             is_part_time,
#             is_full_time,
#             is_informal_care,
#             experience,
#             experience_squared,
#             # high_educ,
#             mother_health,
#             has_sibling,
#         ),
#     )

#     idx = {
#         "agent": 0,
#         "period": 1,
#         "lagged_choice": 2,
#         "choice": 3,
#         "age": 4,
#         "age_squared": 5,
#         "mother_age": 6,
#         "wealth": 7,
#         "savings_rate": 8,
#         "wage": 9,
#         "working_hours": 10,
#         "income": 11,
#         "choice_part_time": 12,
#         "choice_full_time": 13,
#         "choice_informal_care": 14,
#         "experience": 15,
#         "experience_squared": 16,
#         # "high_educ": 17,
#         "mother_health": 17,
#         "has_sibling": 18,
#     }

#     return arr, idx


# def create_simulation_df_from_dict(sim_dict):
#     n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

#     keys_to_drop = ["taste_shocks", "value_choice"]
#     dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

#     return pd.DataFrame(
#         {key: val.ravel() for key, val in dict_to_df.items()},
#         index=pd.MultiIndex.from_product(
#             [np.arange(n_periods), np.arange(n_agents)],
#             names=["period", "agent"],
#         ),
#     )


# def _assign_working_hours_vectorized(choices):
#     no_work_mask = jnp.isin(choices, NOT_WORKING)
#     part_time_mask = jnp.isin(choices, PART_TIME)
#     full_time_mask = jnp.isin(choices, FULL_TIME)

#     return jnp.where(
#         no_work_mask,
#         0,
#         jnp.where(part_time_mask, 20, jnp.where(full_time_mask, 40, jnp.nan)),
#     )
