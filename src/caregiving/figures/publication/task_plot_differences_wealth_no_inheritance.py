"""Publication plot: wealth differences for no-inheritance scenarios."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.model.shared import INFORMAL_CARE


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_wealth_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "wealth_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in assets at period start between two scenarios."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "assets_begin_of_period",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_consumption_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "consumption_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in consumption between the two no-inheritance scenarios."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "consumption",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in consumption (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_savings_dec_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_dec_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in savings decisions between the two scenarios."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "savings_dec",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in savings decision (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_savings_rate_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_rate_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in savings rates between the two no-inheritance scenarios."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "savings_rate",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in savings rate (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_total_income_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "total_income_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in total income between the two scenarios."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "total_income",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in total income (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_accumulated_consumption_difference_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_consumption_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot accumulated consumption difference (baseline minus no-care)."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "consumption",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
        accumulated=True,
        ylabel="Accumulated consumption difference (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_accumulated_savings_dec_difference_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_savings_dec_difference_no_inheritance.pdf",
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot accumulated savings decision difference (baseline minus no-care)."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "savings_dec",
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
        age_min=age_min,
        age_max=age_max,
        accumulated=True,
        ylabel="Accumulated savings decision difference (baseline − no-care)",
    )


# --- Ever-caregiver sample: separate difference plots ---


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_wealth_no_inheritance_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "wealth_difference_no_inheritance_ever_caregivers.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in assets (wealth) for ever-caregiver sample only."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "assets_begin_of_period",
        same_agents=same_agents,
        ever_caregiver=True,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_savings_rate_no_inheritance_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_rate_difference_no_inheritance_ever_caregivers.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in savings rate for ever-caregiver sample only."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "savings_rate",
        same_agents=same_agents,
        ever_caregiver=True,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in savings rate (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_savings_dec_no_inheritance_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_dec_difference_no_inheritance_ever_caregivers.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in savings decision for ever-caregiver sample only."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "savings_dec",
        same_agents=same_agents,
        ever_caregiver=True,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in savings decision (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_accumulated_consumption_difference_no_inheritance_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_consumption_difference_no_inheritance_ever_caregivers.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot accumulated consumption difference for ever-caregiver sample only."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "consumption",
        same_agents=same_agents,
        ever_caregiver=True,
        age_min=age_min,
        age_max=age_max,
        accumulated=True,
        ylabel="Accumulated consumption difference (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_accumulated_savings_dec_difference_no_inheritance_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_savings_dec_difference_no_inheritance_ever_caregivers.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot accumulated savings decision difference for ever-caregiver sample only."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "savings_dec",
        same_agents=same_agents,
        ever_caregiver=True,
        age_min=age_min,
        age_max=age_max,
        accumulated=True,
        ylabel="Accumulated savings decision difference (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_differences_total_income_no_inheritance_ever_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "total_income_difference_no_inheritance_ever_caregivers.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot differences in total income for ever-caregiver sample only."""
    _run_difference_plot(
        path_to_specs,
        path_to_baseline_sim_data,
        path_to_no_care_sim_data,
        path_to_save_plot,
        "total_income",
        same_agents=same_agents,
        ever_caregiver=True,
        age_min=age_min,
        age_max=age_max,
        ylabel="Difference in total income (baseline − no-care)",
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_savings_rate_by_age_baseline_and_no_care_demand_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_rate_by_age_baseline_and_no_care_demand_no_inheritance.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot average savings rate by age for baseline and no care demand.

    Two lines, same agents.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    series_baseline, series_no_care = _build_mean_by_age_both_scenarios(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="savings_rate",
        same_agents=same_agents,
        ever_care_demand=False,
        caregiving_type=False,
        ever_caregiver=False,
    )

    _plot_two_levels_by_age(
        series_baseline=series_baseline,
        series_no_care=series_no_care,
        path_to_save=path_to_save_plot,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_savings_rate_by_age_baseline_and_no_care_demand_no_inheritance_ever_care_demand(  # noqa: E501
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / (
        "savings_rate_by_age_baseline_and_no_care_demand_no_inheritance_"
        "ever_care_demand.pdf"
    ),
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot average savings rate by age for baseline and no care demand.

    Two lines, ever-care-demand sample.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    series_baseline, series_no_care = _build_mean_by_age_both_scenarios(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="savings_rate",
        same_agents=same_agents,
        ever_care_demand=True,
        caregiving_type=False,
        ever_caregiver=False,
    )

    _plot_two_levels_by_age(
        series_baseline=series_baseline,
        series_no_care=series_no_care,
        path_to_save=path_to_save_plot,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_savings_rate_by_age_baseline_and_no_care_demand_no_inheritance_ever_caregivers(  # noqa: E501
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / (
        "savings_rate_by_age_baseline_and_no_care_demand_no_inheritance_"
        "ever_caregivers.pdf"
    ),
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot average savings rate by age for baseline and no care demand.

    Two lines, ever-caregiver sample.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    series_baseline, series_no_care = _build_mean_by_age_both_scenarios(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="savings_rate",
        same_agents=same_agents,
        ever_care_demand=False,
        caregiving_type=False,
        ever_caregiver=True,
    )

    _plot_two_levels_by_age(
        series_baseline=series_baseline,
        series_no_care=series_no_care,
        path_to_save=path_to_save_plot,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_wealth_and_accumulated_savings_dec_difference_no_inheritance(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "wealth_and_accumulated_savings_dec_difference_no_inheritance.pdf",
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot wealth and accumulated savings_dec difference (two lines, same agents)."""
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    diff_wealth = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="assets_begin_of_period",
        same_agents=same_agents,
        ever_care_demand=False,
        caregiving_type=False,
        ever_caregiver=False,
    )
    diff_savings_dec = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="savings_dec",
        same_agents=same_agents,
        ever_care_demand=False,
        caregiving_type=False,
        ever_caregiver=False,
    )
    accumulated_savings_dec = diff_savings_dec.fillna(0).cumsum()

    _plot_two_differences_by_age(
        series_wealth=diff_wealth,
        series_accumulated_savings_dec=accumulated_savings_dec,
        path_to_save=path_to_save_plot,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_wealth_and_accumulated_savings_dec_difference_no_inheritance_ever_caregivers(  # noqa: E501
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / (
        "wealth_and_accumulated_savings_dec_difference_no_inheritance_"
        "ever_caregivers.pdf"
    ),
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot wealth and accumulated savings_dec difference.

    Two lines, ever-caregiver sample.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    diff_wealth = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="assets_begin_of_period",
        same_agents=same_agents,
        ever_care_demand=False,
        caregiving_type=False,
        ever_caregiver=True,
    )
    diff_savings_dec = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="savings_dec",
        same_agents=same_agents,
        ever_care_demand=False,
        caregiving_type=False,
        ever_caregiver=True,
    )
    accumulated_savings_dec = diff_savings_dec.fillna(0).cumsum()

    _plot_two_differences_by_age(
        series_wealth=diff_wealth,
        series_accumulated_savings_dec=accumulated_savings_dec,
        path_to_save=path_to_save_plot,
        age_min=age_min,
        age_max=age_max,
    )


@pytask.mark.publication
@pytask.mark.publication_wealth
def task_plot_wealth_and_accumulated_savings_dec_difference_no_inheritance_ever_care_demand(  # noqa: E501
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_no_care_sim_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand_no_inheritance.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / (
        "wealth_and_accumulated_savings_dec_difference_no_inheritance_"
        "ever_care_demand.pdf"
    ),
    same_agents: bool = True,
    age_min: int = 30,
    age_max: int = 90,
) -> None:
    """Plot wealth and accumulated savings_dec difference.

    Two lines, ever-care-demand sample.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    diff_wealth = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="assets_begin_of_period",
        same_agents=same_agents,
        ever_care_demand=True,
        caregiving_type=False,
        ever_caregiver=False,
    )
    diff_savings_dec = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable="savings_dec",
        same_agents=same_agents,
        ever_care_demand=True,
        caregiving_type=False,
        ever_caregiver=False,
    )
    accumulated_savings_dec = diff_savings_dec.fillna(0).cumsum()

    _plot_two_differences_by_age(
        series_wealth=diff_wealth,
        series_accumulated_savings_dec=accumulated_savings_dec,
        path_to_save=path_to_save_plot,
        age_min=age_min,
        age_max=age_max,
    )


def _run_difference_plot(
    path_to_specs: Path,
    path_to_baseline_sim_data: Path,
    path_to_no_care_sim_data: Path,
    path_to_save_plot: Path,
    variable: str,
    *,
    same_agents: bool = True,
    ever_care_demand: bool = False,
    caregiving_type: bool = False,
    ever_caregiver: bool = False,
    age_min: int = 30,
    age_max: int = 90,
    accumulated: bool = False,
    ylabel: str | None = None,
) -> None:
    """Load data, compute mean difference by age, optionally accumulate, and plot."""
    specs = pickle.load(path_to_specs.open("rb"))
    df_baseline = pd.read_pickle(path_to_baseline_sim_data)
    df_nocare = pd.read_pickle(path_to_no_care_sim_data)

    diff_series = _build_mean_difference_by_age(
        df_baseline=df_baseline,
        df_no_care=df_nocare,
        specs=specs,
        age_min=age_min,
        age_max=age_max,
        variable=variable,
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
    )

    if accumulated:
        diff_series = diff_series.fillna(0).cumsum()

    _plot_difference_by_age(
        diff_series=diff_series,
        age_min=age_min,
        age_max=age_max,
        path_to_save=path_to_save_plot,
        ylabel=ylabel or "Difference in assets at period start (baseline − no-care)",
    )


def _build_mean_difference_by_age(
    df_baseline: pd.DataFrame,
    df_no_care: pd.DataFrame,
    specs: dict,
    age_min: int,
    age_max: int,
    variable: str,
    same_agents: bool,
    ever_care_demand: bool,
    caregiving_type: bool,
    ever_caregiver: bool,
) -> pd.Series:
    """Return baseline minus no-care mean values by age."""

    df_baseline, df_no_care = _prepare_and_filter_sim_dataframes(
        df_baseline,
        df_no_care,
        specs=specs,
        variable=variable,
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
    )

    mean_baseline = df_baseline.groupby("age", observed=False)[variable].mean()
    mean_no_care = df_no_care.groupby("age", observed=False)[variable].mean()

    ages = np.arange(age_min, age_max + 1)
    diff = mean_baseline.sub(mean_no_care, fill_value=0.0).reindex(
        ages, fill_value=np.nan
    )
    return diff


def _build_mean_by_age_both_scenarios(
    df_baseline: pd.DataFrame,
    df_no_care: pd.DataFrame,
    specs: dict,
    age_min: int,
    age_max: int,
    variable: str,
    same_agents: bool,
    ever_care_demand: bool,
    caregiving_type: bool,
    ever_caregiver: bool,
) -> tuple[pd.Series, pd.Series]:
    """Return (mean baseline by age, mean no-care by age) for variable and filter."""
    df_baseline, df_no_care = _prepare_and_filter_sim_dataframes(
        df_baseline,
        df_no_care,
        specs=specs,
        variable=variable,
        same_agents=same_agents,
        ever_care_demand=ever_care_demand,
        caregiving_type=caregiving_type,
        ever_caregiver=ever_caregiver,
    )

    mean_baseline = df_baseline.groupby("age", observed=False)[variable].mean()
    mean_no_care = df_no_care.groupby("age", observed=False)[variable].mean()

    ages = np.arange(age_min, age_max + 1)
    series_baseline = mean_baseline.reindex(ages, fill_value=np.nan)
    series_no_care = mean_no_care.reindex(ages, fill_value=np.nan)
    return series_baseline, series_no_care


def _prepare_and_filter_sim_dataframes(
    df_baseline: pd.DataFrame,
    df_no_care: pd.DataFrame,
    *,
    specs: dict,
    variable: str,
    same_agents: bool,
    ever_care_demand: bool,
    caregiving_type: bool,
    ever_caregiver: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and filter the simulation dataframes according to user options."""

    df_baseline = _prepare_simulation_dataframe(df_baseline, specs, variable)
    df_no_care = _prepare_simulation_dataframe(df_no_care, specs, variable)

    df_baseline = ensure_agent_period(df_baseline).reset_index(drop=True)
    df_no_care = ensure_agent_period(df_no_care).reset_index(drop=True)

    force_same_agents = same_agents or ever_caregiver
    allowed_agents: np.ndarray | None = None
    allowed_agents: np.ndarray | None = None

    if caregiving_type and "caregiving_type" not in df_baseline.columns:
        raise KeyError("Simulation dataframe missing 'caregiving_type' column.")
    if ever_care_demand and "care_demand" not in df_baseline.columns:
        raise KeyError("Baseline simulation dataframe missing 'care_demand' column.")
    if ever_caregiver and "choice" not in df_baseline.columns:
        raise KeyError(
            "Baseline simulation dataframe missing 'choice' column needed for care."
        )

    if caregiving_type:
        ids = df_baseline.loc[df_baseline["caregiving_type"] == 1, "agent"].unique()
        allowed_agents = (
            ids if allowed_agents is None else np.intersect1d(allowed_agents, ids)
        )

    if ever_care_demand:
        ids = df_baseline.loc[df_baseline["care_demand"] > 0, "agent"].unique()
        allowed_agents = (
            ids if allowed_agents is None else np.intersect1d(allowed_agents, ids)
        )

    if ever_caregiver:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        ids = df_baseline.loc[df_baseline["choice"].isin(care_codes), "agent"].unique()
        allowed_agents = (
            ids if allowed_agents is None else np.intersect1d(allowed_agents, ids)
        )

    if allowed_agents is not None:
        df_baseline = df_baseline[df_baseline["agent"].isin(allowed_agents)].copy()

    if allowed_agents is not None:
        df_no_care_agents = df_no_care["agent"].unique()
        common_agents = np.intersect1d(allowed_agents, df_no_care_agents)
        if len(common_agents) < len(allowed_agents):
            missing = np.setdiff1d(allowed_agents, common_agents)
            print(
                f"Warning: {len(missing)} agents from baseline filters "
                "missing in no-care data; dropping them."
            )
        df_baseline = df_baseline[df_baseline["agent"].isin(common_agents)].copy()
        df_no_care = df_no_care[df_no_care["agent"].isin(common_agents)].copy()

    if force_same_agents:
        agent_common = np.intersect1d(
            df_baseline["agent"].unique(), df_no_care["agent"].unique()
        )
        df_baseline = df_baseline[df_baseline["agent"].isin(agent_common)].copy()
        df_no_care = df_no_care[df_no_care["agent"].isin(agent_common)].copy()

        keys_baseline = df_baseline[["agent", "period"]].drop_duplicates()
        keys_nocare = df_no_care[["agent", "period"]].drop_duplicates()
        common_keys = keys_baseline.merge(
            keys_nocare, on=["agent", "period"], how="inner"
        )
        df_baseline = df_baseline.merge(
            common_keys, on=["agent", "period"], how="inner"
        )
        df_no_care = df_no_care.merge(common_keys, on=["agent", "period"], how="inner")

    return df_baseline.reset_index(drop=True), df_no_care.reset_index(drop=True)


def _prepare_simulation_dataframe(
    df: pd.DataFrame, specs: dict, variable: str
) -> pd.DataFrame:
    """Ensure the simulation dataframe has age column and required values."""

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    if variable not in df.columns:
        raise KeyError(f"Simulation dataframe missing '{variable}' column.")
    if "age" not in df.columns:
        if "period" not in df.columns:
            raise ValueError("Simulation data must include 'period' to construct age.")
        df["age"] = df["period"] + specs["start_age"]
    return df


def _plot_difference_by_age(
    diff_series: pd.Series,
    age_min: int,
    age_max: int,
    path_to_save: Path,
    ylabel: str = "Difference in assets at period start (baseline − no-care)",
) -> None:
    """Create the publication-styled line plot of scenario differences."""

    ages = diff_series.index.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(0, color="0.6", linewidth=1.2, linestyle="--")
    ax.plot(
        ages,
        diff_series,
        color="0.15",
        linewidth=2.0,
        linestyle="-",
    )

    ax.set_xlim(age_min - 1, age_max + 1)
    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_save}")


def _plot_two_differences_by_age(
    series_wealth: pd.Series,
    series_accumulated_savings_dec: pd.Series,
    path_to_save: Path,
    age_min: int,
    age_max: int,
) -> None:
    """Create a plot: wealth difference and accumulated savings_dec difference.

    Two lines.
    """

    ages = series_wealth.index.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(0, color="0.6", linewidth=1.2, linestyle="--")
    ax.plot(
        ages,
        series_wealth,
        color="0.15",
        linewidth=2.0,
        linestyle="-",
        label="Difference in wealth (baseline − no-care)",
    )
    ax.plot(
        ages,
        series_accumulated_savings_dec,
        color="0.45",
        linewidth=2.0,
        linestyle="-",
        label="Accumulated savings decision difference (baseline − no-care)",
    )

    ax.set_xlim(age_min - 1, age_max + 1)
    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (baseline − no-care)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8)
    ax.legend(loc="best", fontsize=12)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_save}")


def _plot_two_levels_by_age(
    series_baseline: pd.Series,
    series_no_care: pd.Series,
    path_to_save: Path,
    age_min: int,
    age_max: int,
    ylabel: str = "Average savings rate",
) -> None:
    """Create a plot: mean baseline and mean no-care by age (levels, not difference)."""
    ages = series_baseline.index.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        ages,
        series_baseline,
        color="0.15",
        linewidth=2.0,
        linestyle="-",
        label="Baseline",
    )
    ax.plot(
        ages,
        series_no_care,
        color="0.45",
        linewidth=2.0,
        linestyle="-",
        label="No care demand",
    )

    ax.set_xlim(age_min - 1, age_max + 1)
    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8)
    ax.legend(loc="best", fontsize=12)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_save}")
