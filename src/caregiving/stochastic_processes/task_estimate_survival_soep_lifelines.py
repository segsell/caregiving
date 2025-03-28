# """Estimate the mortality matrix via lifelines package."""

# import itertools
# from pathlib import Path
# from typing import Annotated

# import estimagic as em
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import optimagic as om
# import pandas as pd
# import pytask
# import statsmodels.api as sm
# from pytask import Product

# from caregiving.config import BLD, JET_COLOR_MAP, SRC
# from caregiving.specs.derive_specs import read_and_derive_specs
# from caregiving.stochastic_processes.auxiliary import loglike

# from lifelines import KaplanMeierFitter


# from lifelines.fitters import ParametricUnivariateFitter


# class GompertzFitter(ParametricUnivariateFitter):
#     # this parameterization is slightly different than wikipedia.
#     _fitted_parameter_names = ["nu_", "b_"]

#     def _cumulative_hazard(self, params, times):
#         nu_, b_ = params
#         return nu_ * (np.expm1(times * b_))


# def task_lifelines(
#     path_to_specs: Path = SRC / "specs.yaml",
#     path_to_lifetable: Path = SRC
#     / "data"
#     / "statistical_office"
#     / "mortality_table_for_pandas.csv",
#     path_to_mortatility_sample: Path = BLD
#     / "data"
#     / "mortality_transition_estimation_sample_duplicated.pkl",
#     #     path_to_save_mortality_params_men: Annotated[Path, Product] = BLD
#     #     / "estimation"
#     #     / "stochastic_processes"
#     #     / "mortality_params_three_states_men.csv",
#     #     path_to_save_mortality_params_women: Annotated[Path, Product] = BLD
#     #     / "estimation"
#     #     / "stochastic_processes"
#     #     / "mortality_params_three_states_women.csv",
#     #     path_to_save_mortality_transition_matrix: Annotated[Path, Product] = BLD
#     #     / "estimation"
#     #     / "stochastic_processes"
#     #     / "mortality_transition_matrix_three_states.csv",
#     #     path_to_save_lifetable: Annotated[Path, Product] = BLD
#     #     / "estimation"
#     #     / "stochastic_processes"
#     #     / "lifetable_three_states.csv",
# ):
#     """Estimate the mortality matrix."""

#     specs = read_and_derive_specs(path_to_specs)

#     # Load life table data and expand/duplicate it to include all
#     # possible combinations of health, education and sex
#     lifetable_df = pd.read_csv(
#         path_to_lifetable,
#         sep=";",
#     )

#     mortality_df = pd.DataFrame(
#         [
#             {
#                 "age": row["age"],
#                 "health": combo[0],
#                 # "education": combo[1],
#                 "sex": combo[1],
#                 "death_prob": (
#                     row["death_prob_male"]
#                     if combo[1] == 0
#                     else row["death_prob_female"]
#                 ),  # male (0) or female (1) death prob
#             }
#             for _, row in lifetable_df.iterrows()
#             for combo in list(
#                 itertools.product([0, 1], repeat=2)
#             )  # (health, education, sex)
#         ]
#     )
#     mortality_df.reset_index(drop=True, inplace=True)

#     # Plain life table data
#     lifetable_df = mortality_df[["age", "sex", "death_prob"]].copy()
#     lifetable_df.drop_duplicates(inplace=True)

#     # Estimation sample - as in Kroll Lampert 2008 / Haan Schaller et al. 2024
#     df = pd.read_pickle(path_to_mortatility_sample)

#     # Only keep true sample
#     df = df[df.index.get_level_values("true_sample") == 1].copy()

#     # Sort and remove observations after death
#     # df = df.sort_values(["pid", "syear"])
#     # df["death_cumsum"] = df.groupby("pid")["death event"].cumsum()
#     # df_clean = df[df["death_cumsum"] <= 1].copy()
#     # df_clean.drop(columns=["death_cumsum"], inplace=True)

#     # Sort data by pid and year
#     df = df.sort_values(["pid", "syear"])

#     # Identify first year of death event per individual (if any)
#     df["death_cumsum"] = df.groupby("pid")["death event"].cumsum()

#     # Keep only observations before or at the death event
#     df_clean = df[df["death_cumsum"] <= 1].copy()

#     # Drop helper column
#     df_clean.drop(columns=["death_cumsum"], inplace=True)

#     assert (
#         df_clean.groupby("pid")["death event"].sum().max() <= 1
#     ), "Multiple death events detected!"
#     assert (
#         df_clean.groupby("pid")["death event"].cumsum() <= 1
#     ).all(), "Observations after death event exist!"

#     # # Non-parametric approach

#     # Verify that education is fixed per pid
#     df_clean["fixed_education"] = df_clean.groupby("pid")["education"].transform(
#         "first"
#     )

#     # df_clean["fixed_education"] = df_clean.groupby("pid")["education"].transform("last")

#     # # Estimate empirical death probabilities
#     # death_prob = (
#     #     df_clean.groupby(["age", "fixed_education", "health"])["death event"]
#     #     .mean()
#     #     .reset_index()
#     #     .rename(columns={"death event": "death_probability"})
#     # )

#     # # Compute survival probabilities
#     # death_prob["survival_probability"] = 1 - death_prob["death_probability"]

#     # sns.lineplot(
#     #     data=death_prob,
#     #     x="age",
#     #     y="survival_probability",
#     #     hue="health",
#     #     style="fixed_education",
#     #     markers=True,
#     #     dashes=False,
#     # )
#     # plt.xlabel("Age")
#     # plt.ylabel("Survival Probability")
#     # plt.title("Non-parametric Survival by Age, Health, and Education")
#     # plt.ylim(0, 1)
#     # plt.grid(True)
#     # plt.show()

#     # Prepare data for logistic regression

#     # Step 1: Find first and last observed year per individual

#     df_clean = df_clean.reset_index().copy()

#     # df_duration = (
#     #     df_clean.groupby("pid")
#     #     .agg(
#     #         start_year=("syear", "min"),
#     #         end_year=("syear", "max"),
#     #         death_occurred=("death event", "max"),  # 1 if death occurred, 0 otherwise
#     #     )
#     #     .reset_index()
#     # )

#     # # Step 2: Compute duration
#     # # If death occurred, duration = year of death - first year observed + 1
#     # # If censored (no death), duration = last year observed - first year + 1
#     # # df_duration["duration"] = df_duration["end_year"] - df_duration["start_year"] + 1

#     # # # Create event_observed variable
#     # # df_duration["event_observed"] = df_duration["death_occurred"]

#     # # # Final dataset for Kaplan-Meier:
#     # # km_data = df_duration[["pid", "duration", "event_observed"]]

#     # ages_entry = df_clean.groupby("pid")["age"].first()
#     # durations = df_clean.groupby("pid")["syear"].transform("count")

#     # death_ages = ages_entry + durations
#     # death_ages = np.minimum(death_ages, 100)  # cap maximum age at 100

#     # # Event observed if death age < 100
#     # event_observed = (death_ages < 100).astype(int)
#     # durations = death_ages - ages_entry  # duration reflecting censoring at age 100

#     # # Create DataFrame
#     # km_data = pd.DataFrame({"duration": durations, "event_observed": event_observed})
#     # km_data_clean = km_data.notnull()

#     # kmf = KaplanMeierFitter()
#     # kmf.fit(km_data_clean["duration"], km_data_clean["event_observed"])

#     # kmf.plot_survival_function()
#     # plt.title("Kaplan-Meier Survival Estimate")
#     # plt.xlabel("Time (years)")
#     # plt.ylabel("Survival Probability")
#     # plt.grid(True)
#     # plt.show()

#     # breakpoint()

#     # Step 1: Find first and last observed year per individual

#     # df_duration = df_clean.copy()
#     # df_duration["age_entry"] = df_clean.groupby("pid")["age"].transform("min")
#     # df_duration["start_year"] = df_clean.groupby("pid")["syear"].transform("min")
#     # df_duration["end_age"] = df_clean.groupby("pid")["age"].transform("max")

#     # # Step 2: Compute duration
#     # # If death occurred, duration = year of death - first year observed + 1
#     # # If censored (no death), duration = last year observed - first year + 1
#     # df_duration["duration"] = df_duration["end_age"] - df_duration["age_entry"] + 1

#     # # Final dataset for Kaplan-Meier:
#     # # km_data = df_duration[["education", "duration", "death event"]]

#     # # Initialize Kaplan-Meier Fitter

#     # kmf = KaplanMeierFitter()

#     # # Empty list to store results
#     # results = []

#     # # Loop over education groups
#     # # for edu_group in [0, 1]:
#     # # mask = df_duration["education"] == edu_group
#     # # kmf.fit(df_duration.loc[mask, "duration"], df_duration.loc[mask, "death event"])
#     # kmf.fit(df_duration["duration"], df_duration["death event"])

#     # # Extract survival function as DataFrame
#     # surv_df = kmf.survival_function_.reset_index()
#     # surv_df.columns = ["duration", "survival_probability"]

#     # # Convert duration back to absolute age by adding minimum entry age (approximation)
#     # # min_age = df_duration.loc[mask, "age_entry"].min()
#     # # min_age = df_duration["age_entry"].min()
#     # surv_df["age"] = df_duration["age"]
#     # # surv_df["education"] = edu_group

#     # # Store results
#     # # results.append(surv_df[["age", "education", "survival_probability"]])
#     # results.append(surv_df[["age", "survival_probability"]])

#     # # Concatenate results into a single DataFrame
#     # survival_df = pd.concat(results, ignore_index=True)

#     # # Show result
#     # print(survival_df.head())

#     # survival_df["cumulative_survival"] = survival_df.sort_values("age")[
#     #     "survival_probability"
#     # ].cumprod()

#     from lifelines import CoxPHFitter

#     # Assuming df_clean is your DataFrame
#     # Rename "death event" to a valid column name (e.g., "death_event")
#     df_clean = df_clean.rename(columns={"death event": "death_event"})

#     # # Inspect the data (optional)
#     # print(df_clean.head())

#     # # Initialize and fit the Cox proportional hazards model
#     # cph = CoxPHFitter()
#     # # Here we use 'syear' as the duration column and 'death_event' as the event indicator.
#     # # The formula "age + sex" models the effect of age and sex on survival.
#     # cph.fit(
#     #     df_clean, duration_col="syear", event_col="death_event", formula="age + sex"
#     # )
#     # cph.print_summary()

#     # For age-specific survival probabilities, use 'age' as the duration variable.
#     # Since age is now the time scale, it should not be included as a covariate.
#     # cph = CoxPHFitter()
#     # cph.fit(df_clean, duration_col="age", event_col="death_event", formula="sex")
#     # cph.print_summary()

#     # # Plot the baseline survival function with age on the x-axis
#     # plt.figure(figsize=(8, 6))
#     # baseline_survival = cph.baseline_survival_
#     # plt.plot(
#     #     baseline_survival.index,
#     #     baseline_survival["baseline survival"],
#     #     label="Baseline",
#     # )
#     # plt.xlim(30, 100)
#     # plt.xlabel("Age")
#     # plt.ylabel("Survival Probability")
#     # plt.title("Age-specific Survival Probability (Baseline)")
#     # plt.legend()
#     # plt.show()

#     # # Alternatively, if you want to compare survival curves by sex:
#     # plt.figure(figsize=(8, 6))
#     # for sex_value in sorted(df_clean["sex"].unique()):
#     #     profile = pd.DataFrame({"sex": [sex_value]})
#     #     surv_func = cph.predict_survival_function(profile)
#     #     plt.plot(surv_func.index, surv_func.values.flatten(), label=f"Sex: {sex_value}")
#     # plt.xlim(30, 100)
#     # plt.xlabel("Age")
#     # plt.ylabel("Survival Probability")
#     # plt.title("Age-specific Survival Probability by Sex")
#     # plt.legend()
#     # plt.show()

#     # Rename the event column if needed.
#     df_clean = df_clean.rename(columns={"death event": "death_event"})

#     # Create a figure for the survival curves.
#     plt.figure(figsize=(10, 6))

#     # Define the age grid for plotting.
#     age_grid = np.arange(30, 101)

#     # Fit a Gompertz model separately for each sex.
#     # (Assuming the 'sex' variable is coded, e.g., 0 and 1)
#     for sex_value in sorted(df_clean["sex"].unique()):
#         group = df_clean[df_clean["sex"] == sex_value]

#         # Instantiate and fit the Gompertz model using age as duration.
#         gf = GompertzFitter()
#         gf.fit(
#             durations=group["age"],
#             event_observed=group["death_event"],
#             label=f"Sex {sex_value}",
#         )

#         # Evaluate the survival function at each age in the grid.
#         # The method 'survival_function_at_times' returns a Series.
#         survival_probs = gf.survival_function_at_times(age_grid)

#         # Plot the survival curve.
#         plt.plot(age_grid, survival_probs, label=f"Sex {sex_value}")

#     # Configure the plot.
#     plt.xlabel("Age")
#     plt.ylabel("Survival Probability")
#     plt.title("Age-specific Survival Probability (Gompertz Model)")
#     plt.xlim(30, 100)
#     plt.legend()
#     plt.show()
#     breakpoint()
