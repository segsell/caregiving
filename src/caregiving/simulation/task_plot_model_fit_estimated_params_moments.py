# """Plot model fit using estimated parameters, based on pre-computed moments.

# This task reads empirical moments from ``moments_full.csv`` and simulated
# moments from the corresponding simulated moments file, and produces plots
# comparing the two.
# """

# import pickle
# from pathlib import Path
# from typing import Annotated

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pytask
# from pytask import Product

# from caregiving.config import BLD, JET_COLOR_MAP, SRC

# from caregiving.estimation.prepare_estimation import (
#     load_and_setup_full_model_for_solution,
# )
# from caregiving.model.shared import SCALE_CAREGIVER_SHARE


# @pytask.mark.model_fit_estimated_params_moments
# def task_plot_model_fit_estimated_params_moments(  # noqa: PLR0915
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_empirical_moments: Path = BLD / "moments" / "moments_full.csv",
#     path_to_simulated_moments: Path = BLD
#     / "moments"
#     / "simulated_moments_pandas_estimated_params.csv",
#     path_to_save_labor_shares_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_estimated_params"
#     / "model_fit_from_moments"
#     / "labor_shares_from_moments.png",
#     path_to_save_labor_shares_with_caregivers_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_estimated_params"
#     / "model_fit_from_moments"
#     / "labor_shares_with_caregivers_from_moments.png",
#     path_to_save_caregiver_shares_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_estimated_params"
#     / "model_fit_from_moments"
#     / "share_caregivers_by_age_bin_from_moments.png",
#     path_to_save_work_transitions_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_estimated_params"
#     / "model_fit_from_moments"
#     / "work_transitions_by_edu_and_age_bin_from_moments.png",
#     path_to_save_caregiving_transitions_age_bin_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "model_fit_estimated_params"
#     / "model_fit_from_moments"
#     / "caregiving_transitions_by_age_bin_from_moments.png",
# ) -> None:
#     """Plot labor supply model fit based purely on empirical and simulated moments."""

#     # -------------------------------------------------------------------------
#     # Load specs and moments
#     # -------------------------------------------------------------------------
#     options = pickle.load(path_to_options.open("rb"))

#     model_full = load_and_setup_full_model_for_solution(
#         options, path_to_model=path_to_solution_model
#     )
#     specs = model_full["options"]["model_params"]

#     moms_emp = pd.read_csv(path_to_empirical_moments, index_col=0).squeeze("columns")
#     moms_sim = pd.read_csv(path_to_simulated_moments, index_col=0).squeeze("columns")

#     # -------------------------------------------------------------------------
#     # Plots: labor shares by age and education (from moments)
#     # -------------------------------------------------------------------------
#     plot_labor_moments_by_education_from_moments(
#         moms_emp=moms_emp,
#         moms_sim=moms_sim,
#         specs=specs,
#         path_to_save_plot=path_to_save_labor_shares_plot,
#         include_caregivers=False,
#     )

#     plot_labor_moments_by_education_from_moments(
#         moms_emp=moms_emp,
#         moms_sim=moms_sim,
#         specs=specs,
#         path_to_save_plot=path_to_save_labor_shares_with_caregivers_plot,
#         include_caregivers=True,
#     )

#     # -------------------------------------------------------------------------
#     # Caregiver shares by age bin (overall)
#     # -------------------------------------------------------------------------
#     plot_caregiver_shares_by_age_bin_from_moments(
#         moms_emp=moms_emp,
#         moms_sim=moms_sim,
#         path_to_save_plot=path_to_save_caregiver_shares_age_bin_plot,
#         scale=SCALE_CAREGIVER_SHARE,
#     )

#     # -------------------------------------------------------------------------
#     # Work and caregiving transitions by age bin (from moments)
#     # -------------------------------------------------------------------------
#     plot_work_transitions_by_age_bin_from_moments(
#         moms_emp=moms_emp,
#         moms_sim=moms_sim,
#         specs=specs,
#         path_to_save_plot=path_to_save_work_transitions_age_bin_plot,
#     )

#     plot_caregiving_transitions_by_age_bin_from_moments(
#         moms_emp=moms_emp,
#         moms_sim=moms_sim,
#         path_to_save_plot=path_to_save_caregiving_transitions_age_bin_plot,
#     )


# def plot_labor_moments_by_education_from_moments(
#     moms_emp: pd.Series,
#     moms_sim: pd.Series,
#     specs: dict,
#     path_to_save_plot: Path | None = None,
#     include_caregivers: bool = False,
# ) -> None:
#     """Plot labor supply shares by age and education using moment series.

#     The function expects keys of the form

#     - ``share_<state>_<educ_label>_age_<age>``
#     - ``share_<state>_caregivers_<educ_label>_age_<age>`` (if caregivers used)

#     where ``<educ_label>`` matches lower-cased, underscore-separated
#     entries from ``specs["education_labels"]``.
#     """

#     # States in the order of specs["choice_labels"]
#     choices = ["retired", "unemployed", "part_time", "full_time"]

#     # One row per education group in every figure.
#     # If include_caregivers is False, use non-caregiver moments;
#     # if True, use caregiver-specific moments only.
#     n_rows = 2
#     fig, axs = plt.subplots(
#         n_rows,
#         4,
#         figsize=(16, 6 * (n_rows / 2)),
#         sharex=True,
#         sharey=True,
#     )

#     if n_rows == 1:
#         axs = axs.reshape(1, -1)

#     for _edu_idx, edu_label in enumerate(specs["education_labels"]):
#         edu_token = str(edu_label).lower().replace(" ", "_")

#         row_idx = _edu_idx

#         # ------------------------------------------------------------------
#         # Either general (non-caregiver) or caregiver labor shares
#         # ------------------------------------------------------------------
#         for choice_idx, choice_label in enumerate(specs["choice_labels"]):
#             ax = axs[row_idx, choice_idx]

#             if not include_caregivers:
#                 # Non-caregiver moments: keys do not contain 'caregivers'
#                 emp_keys = [
#                     k
#                     for k in moms_emp.index
#                     if k.startswith(f"share_{choices[choice_idx]}_")
#                     and edu_token in k
#                     and "caregivers" not in k
#                 ]
#                 sim_keys = [
#                     k
#                     for k in moms_sim.index
#                     if k.startswith(f"share_{choices[choice_idx]}_")
#                     and edu_token in k
#                     and "caregivers" not in k
#                 ]
#             else:
#                 # Caregiver-specific moments only
#                 emp_keys = [
#                     k
#                     for k in moms_emp.index
#                     if k.startswith(f"share_{choices[choice_idx]}_caregivers_")
#                     and edu_token in k
#                 ]
#                 sim_keys = [
#                     k
#                     for k in moms_sim.index
#                     if k.startswith(f"share_{choices[choice_idx]}_caregivers_")
#                     and edu_token in k
#                 ]

#             emp_keys_sorted = sorted(emp_keys, key=_extract_age_from_key)
#             sim_keys_sorted = sorted(sim_keys, key=_extract_age_from_key)

#             emp_ages = [_extract_age_from_key(k) for k in emp_keys_sorted]
#             emp_values = [moms_emp[k] for k in emp_keys_sorted]
#             sim_ages = [_extract_age_from_key(k) for k in sim_keys_sorted]
#             sim_values = [moms_sim[k] for k in sim_keys_sorted]

#             ax.plot(sim_ages, sim_values, label="Simulated", color=JET_COLOR_MAP[0])
#             ax.plot(
#                 emp_ages,
#                 emp_values,
#                 label="Observed",
#                 ls="--",
#                 color=JET_COLOR_MAP[1],
#             )

#             ax.set_xlabel("Age")
#             ax.set_ylim([0, 1])
#             title_suffix = " (caregivers)" if include_caregivers else ""
#             ax.set_title(f"{choice_label} - {edu_label}{title_suffix}")
#             ax.tick_params(labelbottom=True)

#             if choice_idx == 0:
#                 ax.set_ylabel("Share")
#                 ax.legend()
#             else:
#                 ax.set_ylabel("")

#     fig.tight_layout()

#     if path_to_save_plot is not None:
#         path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(path_to_save_plot, dpi=300)

#     plt.close(fig)


# def _extract_age_from_key(key: str) -> int:
#     """Extract trailing age (last underscore-separated integer) from a key."""

#     return int(key.split("_")[-1])


# def plot_caregiver_shares_by_age_bin_from_moments(
#     moms_emp: pd.Series,
#     moms_sim: pd.Series,
#     path_to_save_plot: Path | None = None,
#     scale: float = 1.0,
# ) -> None:
#     """Plot share of informal caregivers by age bin.

#     Uses empirical and simulated moments.
#     """

#     prefix = "share_informal_care_age_bin_"

#     emp_keys = [k for k in moms_emp.index if k.startswith(prefix)]
#     sim_keys = [k for k in moms_sim.index if k.startswith(prefix)]

#     def _bin_start(key: str) -> int:
#         # key format: share_informal_care_age_bin_<start>_<end>
#         parts = key.split("_")
#         return int(parts[-2])

#     emp_keys_sorted = sorted(emp_keys, key=_bin_start)
#     sim_keys_sorted = sorted(sim_keys, key=_bin_start)

#     bin_starts_emp = np.array([_bin_start(k) for k in emp_keys_sorted])
#     emp_values = np.array([moms_emp[k] for k in emp_keys_sorted]) / scale

#     bin_starts_sim = np.array([_bin_start(k) for k in sim_keys_sorted])
#     sim_values = np.array([moms_sim[k] for k in sim_keys_sorted]) / scale

#     # Align bins present in both empirical and simulated and drop 75–79 bin
#     common_bins = np.intersect1d(bin_starts_emp, bin_starts_sim)
#     # match main plot that drops 75–79
#     max_age_threshold = 75
#     keep_mask = common_bins < max_age_threshold
#     bin_starts = common_bins[keep_mask]
#     emp_values = emp_values[np.isin(bin_starts_emp, bin_starts)]
#     sim_values = sim_values[np.isin(bin_starts_sim, bin_starts)]

#     fig, ax = plt.subplots(figsize=(8, 4))

#     bar_w = 1.5
#     gap = 0.10
#     offset = bar_w / 2 + gap / 2

#     x_emp = bin_starts - offset
#     x_sim = bin_starts + offset

#     ax.bar(x_emp, emp_values, width=bar_w, color=JET_COLOR_MAP[1], label="Observed")
#     ax.bar(x_sim, sim_values, width=bar_w, color=JET_COLOR_MAP[0], label="Simulated")

#     xticks = bin_starts
#     # approximating bin_width from consecutive starts if possible
#     if len(bin_starts) > 1:
#         approx_bin_width = bin_starts[1] - bin_starts[0]
#     else:
#         approx_bin_width = 5
#     xticklabels = [
#         f"{start}\u2013{start + approx_bin_width - 1}" for start in bin_starts
#     ]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels)

#     ax.set_xlabel("Age group")
#     ax.set_ylabel("Share of informal caregivers")
#     ax.legend()
#     ax.set_xlim(bin_starts[0] - bar_w - gap, bin_starts[-1] + bar_w + gap)
#     ax.set_ylim(0, 0.12)

#     fig.tight_layout()

#     if path_to_save_plot is not None:
#         path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(path_to_save_plot, dpi=300)

#     plt.close(fig)


# def plot_work_transitions_by_age_bin_from_moments(
#     moms_emp: pd.Series,
#     moms_sim: pd.Series,
#     specs: dict,
#     path_to_save_plot: Path | None = None,
# ) -> None:
#     """Plot work→work transition probabilities by age bin and education."""

#     edu_tokens = ["low_education", "high_education"]
#     edu_labels = specs.get("education_labels", ["Low education", "High education"])

#     fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

#     for idx, edu_token in enumerate(edu_tokens):
#         ax = axs[idx]

#         prefix = f"trans_working_to_working_{edu_token}_age_"
#         emp_keys = [k for k in moms_emp.index if k.startswith(prefix)]
#         sim_keys = [k for k in moms_sim.index if k.startswith(prefix)]

#         def _bin_start_from_trans(key: str) -> int:
#             # key format: trans_working_to_working_<edu>_age_<start>_<end>
#             parts = key.split("_")
#             return int(parts[-2])

#         emp_keys_sorted = sorted(emp_keys, key=_bin_start_from_trans)
#         sim_keys_sorted = sorted(sim_keys, key=_bin_start_from_trans)

#         emp_bins = np.array([_bin_start_from_trans(k) for k in emp_keys_sorted])
#         sim_bins = np.array([_bin_start_from_trans(k) for k in sim_keys_sorted])

#         common_bins = np.intersect1d(emp_bins, sim_bins)
#         emp_vals = np.array([moms_emp[k] for k in emp_keys_sorted])[
#             np.isin(emp_bins, common_bins)
#         ]
#         sim_vals = np.array([moms_sim[k] for k in sim_keys_sorted])[
#             np.isin(sim_bins, common_bins)
#         ]

#         ax.plot(common_bins, sim_vals, label="Simulated", color=JET_COLOR_MAP[0])
#         ax.plot(
#             common_bins,
#             emp_vals,
#             label="Observed",
#             color=JET_COLOR_MAP[1],
#             linestyle="--",
#         )

#         ax.set_xlabel("Age (bin start)")
#         if idx == 0:
#             ax.set_ylabel("Pr(Work→Work)")
#             ax.legend()
#         ax.set_title(edu_labels[idx])
#         ax.set_ylim(0, 1)

#     fig.tight_layout()

#     if path_to_save_plot is not None:
#         path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(path_to_save_plot, dpi=300)

#     plt.close(fig)


# def plot_caregiving_transitions_by_age_bin_from_moments(
#     moms_emp: pd.Series,
#     moms_sim: pd.Series,
#     path_to_save_plot: Path | None = None,
# ) -> None:
#     """Plot caregiving→caregiving transition probabilities by age bin.

#     Pooled education.
#     """

#     prefix = "trans_caregiving_to_caregiving_all_education_age_"

#     emp_keys = [k for k in moms_emp.index if k.startswith(prefix)]
#     sim_keys = [k for k in moms_sim.index if k.startswith(prefix)]

#     def _bin_start_from_care(key: str) -> int:
#         # key format: trans_caregiving_to_caregiving_all_education_age_<start>_<end>
#         parts = key.split("_")
#         return int(parts[-2])

#     emp_keys_sorted = sorted(emp_keys, key=_bin_start_from_care)
#     sim_keys_sorted = sorted(sim_keys, key=_bin_start_from_care)

#     emp_bins = np.array([_bin_start_from_care(k) for k in emp_keys_sorted])
#     sim_bins = np.array([_bin_start_from_care(k) for k in sim_keys_sorted])

#     common_bins = np.intersect1d(emp_bins, sim_bins)
#     emp_vals = np.array([moms_emp[k] for k in emp_keys_sorted])[
#         np.isin(emp_bins, common_bins)
#     ]
#     sim_vals = np.array([moms_sim[k] for k in sim_keys_sorted])[
#         np.isin(sim_bins, common_bins)
#     ]

#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.plot(common_bins, sim_vals, label="Simulated", color=JET_COLOR_MAP[0])
#     ax.plot(
#         common_bins,
#         emp_vals,
#         label="Observed",
#         color=JET_COLOR_MAP[1],
#         linestyle="--",
#     )

#     ax.set_xlabel("Age (bin start)")
#     ax.set_ylabel("Pr(Care→Care)")
#     ax.set_ylim(0, 1)
#     ax.legend()

#     fig.tight_layout()

#     if path_to_save_plot is not None:
#         path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(path_to_save_plot, dpi=300)

#     plt.close(fig)
