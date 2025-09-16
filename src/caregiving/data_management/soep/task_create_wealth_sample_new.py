"""Load and process wealth data."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.task_write_specs import read_and_derive_specs
from caregiving.utils import table


# def task_create_household_wealth_sample(
#     path_to_specs: Path = SRC / "specs.yaml",
#     path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
#     soep_c38_hwealth: Path = SRC / "data" / "soep" / "hwealth.dta",
#     path_to_save: Annotated[Path, Product] = BLD / "data" / "soep_wealth_data.csv",
# ) -> None:
#     """Create sample for wealth estimation."""

#     specs = read_and_derive_specs(path_to_specs)
#     # specs["start_year"] = 2010
#     # specs["end_year"] = 2017

#     cpi = pd.read_csv(path_to_cpi, index_col=0)

#     wealth_data = pd.read_stata(
#         soep_c38_hwealth,
#         columns=["hid", "syear", "w011ha"],
#         convert_categoricals=False,
#     )
#     wealth_data["hid"] = wealth_data["hid"].astype(int)

#     wealth_data = trim_and_rename(wealth_data)
#     wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data, specs)

#     # data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
#     data = deflate_wealth(wealth_data_full, cpi_data=cpi, specs=specs)

#     data.loc[data["wealth"] < 0, "wealth"] = 0
#     # data.set_index(["pid", "syear"], inplace=True)
#     # data = data[(data["wealth"].notna())]

#     print(str(len(data)) + " left after dropping people with missing wealth.")

#     data.to_csv(path_to_save)


# # =====================================================================================
# # Helper functions
# # =====================================================================================


# # hid 167, 302, 930, 981, 1031, 2046, 5240, 9474, 3490091, 3503398 show some edge cases and how they are handled
# def interpolate_and_extrapolate_wealth(wealth_data_full):
#     """Interpolate wealth for each household (consistent hh size) and extrapolate"""

#     # interpolate between existing points with consistent household size (only people specs["start_age"] or older)
#     wealth_data_full["hh_size_adjusted"] = wealth_data_full.groupby(["hid", "syear"])[
#         "pid"
#     ].count()
#     # only keep households with ppathl data for at least one person in a given year
#     wealth_data_full = wealth_data_full[wealth_data_full["hh_size_adjusted"] > 0]
#     # in households with 2+ people above start age drop people with no partner in the household
#     # this removes e.g. - child thats older than start age and living with parents
#     #                   - a person that is not a partner and living with a couple e.g. a retired parent
#     #                   - shared apartment (e.g. two/three/four friends above start age)
#     #                   - separated couple living together in some cases
#     # around 700 household are effected with 7500 observations being dropped (out of 360k)
#     wealth_data_full = wealth_data_full[
#         ~(
#             (wealth_data_full["hh_size_adjusted"] >= 2)
#             & (wealth_data_full["is_par"] == 0)
#         )
#     ]
#     # recalculate household size without the dropped people
#     wealth_data_full["hh_size_adjusted"] = wealth_data_full.groupby(["hid", "syear"])[
#         "pid"
#     ].count()
#     # drop households with more than 2 people above start age left. This happens e.g. if two or more couples live
#     # together or a married person moves in with their parents doing care work etc.
#     # this happens in around 80 households with 700 observations being dropped
#     wealth_data_full = wealth_data_full[wealth_data_full["hh_size_adjusted"] <= 2]

#     # Interpolate wealth between observations
#     wealth_data_full["wealth"] = wealth_data_full.groupby(
#         ["hid", "hh_size_adjusted", "pid"]
#     )["wealth"].transform(
#         lambda group: group.interpolate(method="linear", limit_area="inside")
#     )

#     # Keep hh if the hh (by size and pid) has at least one non-NaN wealth value for a syear, mask == NaN
#     # if is_par == NaN => drop rows, since they are not in the panel
#     mask = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"])[
#         "wealth"
#     ].transform(lambda x: x.notna().any())
#     mask &= wealth_data_full["is_par"].notna()
#     wealth_data_full = wealth_data_full[mask]

#     # extrapolate wealth for each household at the start and end of the panel if there are at least 2 valid observations
#     # (by pid too in case household goes from A to AB to B)
#     extrapolated = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"]).apply(
#         extrapolate_wealth_linear
#     )
#     # merge the extrapolated wealth back into the original dataframe
#     extrapolated = extrapolated.drop(columns=["hid", "hh_size_adjusted", "pid"])
#     extrapolated = extrapolated.reset_index()
#     wealth_data_full = (
#         wealth_data_full.reset_index()
#         .drop(columns="wealth")
#         .merge(
#             extrapolated[["hid", "syear", "wealth"]], on=["hid", "syear"], how="left"
#         )
#         .set_index(["hid", "syear", "pid"])
#     )

#     # drop duplicates caused by doing the extrapolation twice in hh with 2 people
#     wealth_data_full = wealth_data_full[
#         ~wealth_data_full.index.duplicated(keep="first")
#     ]

#     # find mean hh age for each household (rounded to nearest 5 years)
#     wealth_data_full["mean_hh_age_rounded"] = wealth_data_full.groupby(
#         ["hid", "syear"]
#     )["float_age"].transform(lambda x: (x.mean() / 5.0).round(0) * 5)

#     # create a dataframe with percentiles (1% to 100%) of wealth, grouped by each unique combination of: syear, mean_hh_age_rounded, hh_size_adjusted
#     wealth_data_unique = wealth_data_full.reset_index().drop_duplicates(
#         subset=["hid", "syear"]
#     )
#     wealth_deciles_df = (
#         wealth_data_unique.groupby(["syear", "mean_hh_age_rounded", "hh_size_adjusted"])
#         .apply(compute_deciles)
#         .reset_index()
#         .set_index(["syear", "mean_hh_age_rounded", "hh_size_adjusted"])
#     )
#     wealth_deciles_df = wealth_deciles_df.dropna(how="all")

#     # impute wealth for households with only one valid observation
#     imputed = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"]).apply(
#         lambda group: impute_wealth_from_deciles(group, wealth_deciles_df)
#     )

#     # merge the imputed wealth back into the original dataframe
#     imputed = imputed.drop(columns=["hid", "hh_size_adjusted", "pid"])
#     imputed = imputed.reset_index()
#     wealth_data_full = (
#         wealth_data_full.reset_index()
#         .drop(columns="wealth")
#         .merge(imputed[["hid", "syear", "wealth"]], on=["hid", "syear"], how="left")
#         .set_index(["hid", "syear", "pid"])
#     )

#     # drop duplicates caused by doing the imputation twice in hh with 2 people
#     wealth_data_full = wealth_data_full[
#         ~wealth_data_full.index.duplicated(keep="first")
#     ]

#     return wealth_data_full


# def extrapolate_wealth_linear(household):
#     """Linearly extrapolate wealth for a household if at least 2 valid observations exist."""
#     household_int = household.reset_index()
#     wealth = household_int["wealth"].copy()
#     syear = household_int["syear"].copy()
#     valid = wealth.dropna()

#     if len(valid) >= 2:
#         # Linear extrapolation at start
#         if pd.isnull(wealth.iloc[0]):
#             first_valid_index = valid.index[0]
#             missing_start_idxs = wealth.index[wealth.index < first_valid_index]
#             y = valid.iloc[:2].values
#             x = syear.iloc[valid.iloc[:2].index].values
#             slope = (y[1] - y[0]) / (x[1] - x[0])
#             wealth.loc[missing_start_idxs] = y[0] - slope * (
#                 first_valid_index - missing_start_idxs
#             )

#         # Linear extrapolation at end
#         if pd.isnull(wealth.iloc[-1]):
#             last_valid_index = valid.index[-1]
#             missing_end_idxs = wealth.index[wealth.index > last_valid_index]
#             y = valid.iloc[-2:].values
#             x = syear.iloc[valid.iloc[-2:].index].values
#             slope = (y[1] - y[0]) / (x[1] - x[0])
#             wealth.loc[missing_end_idxs] = y[1] + slope * (
#                 missing_end_idxs - last_valid_index
#             )

#     household_int["wealth"] = wealth
#     return household_int.set_index("syear")


# def impute_wealth_from_deciles(household, wealth_deciles_df):
#     """Impute wealth for a household with only one valid observation using deciles."""
#     household_int = household.reset_index()
#     valid = household_int["wealth"].dropna()

#     if len(valid) != 1:
#         return household_int.set_index("syear")

#     valid_index = valid.index[0]
#     row_valid = household_int.loc[valid_index]
#     valid_value = row_valid["wealth"]
#     syear_valid = row_valid["syear"]
#     age_valid = row_valid["mean_hh_age_rounded"]
#     size_valid = row_valid["hh_size_adjusted"]

#     # find percentile approx via linear interpolation between deciles
#     try:
#         deciles = wealth_deciles_df.loc[(syear_valid, age_valid, size_valid)].values
#     except KeyError:
#         # if no deciles exist for the valid wealth observation, leave the wealth as NaN everywhere
#         return household_int.set_index("syear")

#     decile_bounds = np.arange(10, 100, 10) / 100  # 0.1 to 0.9
#     f_x_j = 0.1  # since deciles are 10% intervals

#     # find first class j, such that F(x_{j_o}) >= p
#     below_10th = True
#     above_90th = False
#     for i in range(len(deciles) - 1):
#         if deciles[i] <= valid_value <= deciles[i + 1]:
#             below_10th = False
#             x_j_u = deciles[i]
#             x_j_o = deciles[i + 1]
#             F_x_j_u = decile_bounds[i]
#             # linear interpolation formula for p of valid_value
#             p = (
#                 F_x_j_u + f_x_j * (valid_value - x_j_u) / (x_j_o - x_j_u)
#                 if x_j_u != x_j_o
#                 else F_x_j_u
#             )
#             break
#     else:  # valid wealth above 9th decile, set p to 0.9
#         above_90th = True

#     # impute missing values
#     for idx, row in household_int.iterrows():
#         if pd.notnull(row["wealth"]):
#             continue
#         else:
#             target_key = (
#                 row["syear"],
#                 row["mean_hh_age_rounded"],
#                 row["hh_size_adjusted"],
#             )
#             try:
#                 target_deciles = wealth_deciles_df.loc[target_key].values
#             except KeyError:
#                 # if no deciles exist for the that year leave the wealth as NaN
#                 continue

#             if (
#                 below_10th
#             ):  # if valid wealth was below 10th decile return the 10th decile
#                 wealth_guess = target_deciles[0]
#             elif (
#                 above_90th
#             ):  # if valid wealth was above 90th decile return the 90th decile
#                 wealth_guess = target_deciles[-1]
#             else:
#                 # p and F_x_j_u from previous step
#                 x_j_u = target_deciles[int(F_x_j_u * 10) - 1]
#                 x_j_o = target_deciles[int(F_x_j_u * 10)]
#                 # Impute wealth using the linear interpolation formula
#                 wealth_guess = x_j_u + ((p - F_x_j_u) / f_x_j) * (x_j_o - x_j_u)

#             household_int.loc[idx, "wealth"] = wealth_guess

#     return household_int.set_index("syear")


# def span_full_wealth_panel(wealth_data, specs):
#     """Creates additional rows for each household for each year between start_year and
#     end_year.

#     Every household without any wealth data is dropped.

#     """
#     hid_values = wealth_data.index.get_level_values("hid").unique()
#     min_year_wealth = wealth_data.index.get_level_values("syear").min()
#     max_year_wealth = wealth_data.index.get_level_values("syear").max()
#     start_year = np.minimum(min_year_wealth, specs["start_year"])
#     end_year = np.maximum(max_year_wealth, specs["end_year"])

#     all_index = pd.MultiIndex.from_product(
#         [hid_values, range(start_year, end_year + 1)], names=["hid", "syear"]
#     )
#     wealth_data_full = wealth_data.reindex(all_index, fill_value=np.nan, copy=True)
#     return wealth_data_full


# def compute_deciles(group):
#     if group["wealth"].isna().all():
#         return pd.Series(
#             np.full(9, np.nan),  # 10% to 90% deciles
#             index=[f"w_dcl_{i}" for i in range(1, 10)],
#         )
#     else:
#         return pd.Series(
#             np.nanpercentile(
#                 group["wealth"], np.arange(10, 100, 10)
#             ),  # Dezile 10% bis 90%
#             index=[f"w_dcl_{i}" for i in range(1, 10)],
#         )


# def load_wealth_data(soep_c38_path):
#     # Load SOEP core data
#     wealth_data = pd.read_stata(
#         f"{soep_c38_path}/hwealth.dta",
#         columns=["hid", "syear", "w011ha"],  # "w011hb", "w011hc", "w011hd", "w020h0"
#         convert_categoricals=False,
#     )
#     wealth_data["hid"] = wealth_data["hid"].astype(int)
#     wealth_data.set_index(["hid", "syear"], inplace=True)
#     wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)
#     return wealth_data


# def add_personal_data(path_dict, specs, wealth_data_full, use_processed_pl=True):
#     """Load ppathl data from SOEP."""
#     soep_c38_path = path_dict["soep_c38"]
#     # Start with ppathl. Everyone is in there even if not individually surveyed and just member
#     # of surveyed household
#     ppathl_data = pd.read_stata(
#         f"{soep_c38_path}/ppathl.dta",
#         columns=[
#             "pid",
#             "hid",
#             "syear",
#             "sex",
#             "parid",
#             "gebjahr",
#             "gebmonat",
#         ],  # "rv_id", ...
#         convert_categoricals=False,
#     )
#     ppathl_data.dropna(inplace=True)  # drop if most basic data is missing
#     ppathl_data["hid"] = ppathl_data["hid"].astype(int)

#     pl_intermediate_file = path_dict["intermediate_data"] + "pl_structural_w.pkl"
#     if use_processed_pl:
#         pl_data = pd.read_pickle(pl_intermediate_file)
#     else:
#         # Add pl data
#         pl_data_reader = pd.read_stata(
#             f"{soep_c38_path}/pl.dta",
#             columns=[
#                 "pid",
#                 "hid",
#                 "syear",
#                 "pmonin",
#                 "ptagin",
#             ],  # "plb0304_h", "iyear", ...
#             chunksize=100000,
#             convert_categoricals=False,
#         )
#         pl_data = pd.DataFrame()
#         for itm in pl_data_reader:
#             pl_data = pd.concat([pl_data, itm])

#         pl_data["hid"] = pl_data["hid"].astype(int)
#         pl_data.to_pickle(pl_intermediate_file)

#     merged_data = pd.merge(ppathl_data, pl_data, on=["pid", "syear", "hid"], how="left")

#     # set index to pid and syear, create age and filter by age, drop pids
#     merged_data.set_index(["pid", "syear"], inplace=True)
#     merged_data = calc_age_at_interview(merged_data)
#     merged_data = merged_data[merged_data["age"] >= specs["start_age"]]
#     merged_data["is_par"] = np.where(merged_data["parid"] == -2, 0, 1)
#     merged_data.reset_index(inplace=True)

#     # add number of children in household to merged_data
#     pequiv_data = pd.read_stata(
#         # d11107: number of children in household
#         f"{soep_c38_path}/pequiv.dta",
#         columns=["pid", "syear", "d11107"],
#     )
#     merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
#     merged_data.rename(columns={"d11107": "children"}, inplace=True)
#     merged_data.reset_index(inplace=True)
#     merged_data.set_index(["hid", "syear"], inplace=True)

#     # Merge wealth data with hid/syear information and return
#     return wealth_data_full.merge(
#         merged_data, right_index=True, left_index=True, how="left"
#     )
