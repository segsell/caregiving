"""Plot asset and accumulated savings-dec differences.

Baseline vs no care demand in one figure.
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import PUBLICATION_PLOT_STYLE
from caregiving.model.shared import DEAD


def _extract_aux_variable(df_sim: pd.DataFrame, var_name: str) -> pd.Series:
    """Extract a variable from either direct column or aux dictionary."""
    if var_name in df_sim.columns:
        return df_sim[var_name]
    if "aux" in df_sim.columns:
        return df_sim["aux"].apply(
            lambda x: (x.get(var_name, np.nan) if isinstance(x, dict) else np.nan)
        )
    return pd.Series(np.nan, index=df_sim.index)


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_asset_and_accumulated_savings_dec_differences_combined(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "asset_and_accumulated_savings_dec_differences_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot asset and savings-dec difference by age in one figure.

    Same data as asset_differences_by_age and accumulated_savings_dec_difference_by_age:
    - Black solid line: difference in average assets by age (baseline − no care demand).
    - Dashed grey line: accumulated (cumulative) difference in savings decision by age.

    Style matches the light/intensive care demand by age plots (pre_estimation): font,
    figsize, axis labels, grid, spines, PDF output.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # ----- Asset difference by age -----
    if "assets_begin_of_period" not in df_baseline.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if "assets_begin_of_period" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'assets_begin_of_period' not found in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    avg_assets_baseline = (
        df_baseline.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_baseline.columns = ["age", "avg_assets_baseline"]

    avg_assets_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["assets_begin_of_period"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_assets_no_care_demand.columns = ["age", "avg_assets_no_care_demand"]

    merged_assets = avg_assets_baseline.merge(
        avg_assets_no_care_demand, on="age", how="inner"
    )
    merged_assets["diff_assets"] = (
        merged_assets["avg_assets_baseline"]
        - merged_assets["avg_assets_no_care_demand"]
    )

    # ----- Accumulated savings_dec difference by age -----
    if "savings_dec" not in df_baseline.columns:
        df_baseline["savings_dec"] = _extract_aux_variable(df_baseline, "savings_dec")
    if "savings_dec" not in df_no_care_demand.columns:
        df_no_care_demand["savings_dec"] = _extract_aux_variable(
            df_no_care_demand, "savings_dec"
        )
    if "savings_dec" not in df_baseline.columns:
        raise ValueError(
            "Column 'savings_dec' not found in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if "savings_dec" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'savings_dec' not found in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    avg_savings_dec_baseline = (
        df_baseline.groupby("age", observed=False)["savings_dec"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_savings_dec_baseline.columns = ["age", "avg_savings_dec_baseline"]

    avg_savings_dec_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["savings_dec"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_savings_dec_no_care_demand.columns = ["age", "avg_savings_dec_no_care_demand"]

    merged_savings = avg_savings_dec_baseline.merge(
        avg_savings_dec_no_care_demand, on="age", how="inner"
    )
    merged_savings["diff_savings_dec"] = (
        merged_savings["avg_savings_dec_baseline"]
        - merged_savings["avg_savings_dec_no_care_demand"]
    )
    merged_savings = merged_savings.sort_values("age").reset_index(drop=True)
    merged_savings["accumulated_diff_savings_dec"] = merged_savings[
        "diff_savings_dec"
    ].cumsum()

    # ----- Combine on age -----
    combined = merged_assets[["age", "diff_assets"]].merge(
        merged_savings[["age", "accumulated_diff_savings_dec"]],
        on="age",
        how="inner",
    )
    combined = combined.sort_values("age").reset_index(drop=True)
    combined = combined[combined["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        combined["age"],
        combined["diff_assets"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )
    ax.plot(
        combined["age"],
        combined["accumulated_diff_savings_dec"],
        color="0.5",
        linewidth=linewidth - 0.5,
        linestyle="--",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (1000 EUR)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(
        "Combined asset and accumulated savings-dec "
        f"differences plot saved to {path_to_plot}"
    )


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_savings_rate_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_rate_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot savings rate difference by age (baseline - no care demand).

    One black line.

    savings_rate = savings_dec / total_income (NaN where total_income <= 0).
    Same style as asset/savings-dec combined plot; only ages up to 89, no legend.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "savings_dec" not in df.columns:
            df["savings_dec"] = _extract_aux_variable(df, "savings_dec")
        if "total_income" not in df.columns:
            df["total_income"] = _extract_aux_variable(df, "total_income")

    if (
        "savings_dec" not in df_baseline.columns
        or "total_income" not in df_baseline.columns
    ):
        raise ValueError(
            "Columns 'savings_dec' and 'total_income' required in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if (
        "savings_dec" not in df_no_care_demand.columns
        or "total_income" not in df_no_care_demand.columns
    ):
        raise ValueError(
            "Columns 'savings_dec' and 'total_income' required in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    df_baseline["savings_rate"] = np.where(
        df_baseline["total_income"] > 0,
        df_baseline["savings_dec"] / df_baseline["total_income"],
        np.nan,
    )
    df_no_care_demand["savings_rate"] = np.where(
        df_no_care_demand["total_income"] > 0,
        df_no_care_demand["savings_dec"] / df_no_care_demand["total_income"],
        np.nan,
    )

    avg_sr_baseline = (
        df_baseline.groupby("age", observed=False)["savings_rate"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_sr_baseline.columns = ["age", "avg_savings_rate_baseline"]

    avg_sr_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["savings_rate"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_sr_no_care_demand.columns = ["age", "avg_savings_rate_no_care_demand"]

    merged = avg_sr_baseline.merge(avg_sr_no_care_demand, on="age", how="inner")
    merged["diff_savings_rate"] = (
        merged["avg_savings_rate_baseline"] - merged["avg_savings_rate_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["diff_savings_rate"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (baseline − no care demand)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Savings rate difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_own_income_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "own_income_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot own income difference by age (baseline - no care demand).

    One black line.

    total_income is total own income including labor and pension income.
    Same style as savings rate difference plot; only ages up to 89, no legend.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "total_income" not in df.columns:
            df["total_income"] = _extract_aux_variable(df, "total_income")

    if "total_income" not in df_baseline.columns:
        raise ValueError(
            "Column 'total_income' not found in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if "total_income" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'total_income' not found in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    avg_inc_baseline = (
        df_baseline.groupby("age", observed=False)["total_income"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_inc_baseline.columns = ["age", "avg_own_income_baseline"]

    avg_inc_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["total_income"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_inc_no_care_demand.columns = ["age", "avg_own_income_no_care_demand"]

    merged = avg_inc_baseline.merge(avg_inc_no_care_demand, on="age", how="inner")
    merged["diff_own_income"] = (
        merged["avg_own_income_baseline"] - merged["avg_own_income_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["diff_own_income"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (baseline − no care demand)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Own income (total_income) difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_accumulated_consumption_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_consumption_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot accumulated consumption difference by age.

    Baseline - no care demand, one black line.

    diff(age) = avg_consumption_baseline(age) - avg_consumption_no_care_demand(age);
    accumulated_diff(age) = sum(diff(age') for age' <= age).
    Same style as other difference plots; only ages up to 89, no legend.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "consumption" not in df.columns:
            df["consumption"] = _extract_aux_variable(df, "consumption")

    if "consumption" not in df_baseline.columns:
        raise ValueError(
            "Column 'consumption' not found in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if "consumption" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'consumption' not found in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    avg_cons_baseline = (
        df_baseline.groupby("age", observed=False)["consumption"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_cons_baseline.columns = ["age", "avg_consumption_baseline"]

    avg_cons_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["consumption"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_cons_no_care_demand.columns = ["age", "avg_consumption_no_care_demand"]

    merged = avg_cons_baseline.merge(avg_cons_no_care_demand, on="age", how="inner")
    merged["diff_consumption"] = (
        merged["avg_consumption_baseline"] - merged["avg_consumption_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged["accumulated_diff_consumption"] = merged["diff_consumption"].cumsum()
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["accumulated_diff_consumption"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (1000 EUR)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Accumulated consumption difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_consumption_rate_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "consumption_rate_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot consumption rate difference by age (baseline - no care demand).

    One black line.
    consumption_rate = consumption / total_income (NaN where
    total_income <= 0). Same idea as
    consumption_rate_difference_by_age.png; same style,
    only ages up to 89, no legend.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "consumption" not in df.columns:
            df["consumption"] = _extract_aux_variable(df, "consumption")
        if "total_income" not in df.columns:
            df["total_income"] = _extract_aux_variable(df, "total_income")

    if (
        "consumption" not in df_baseline.columns
        or "total_income" not in df_baseline.columns
    ):
        raise ValueError(
            "Columns 'consumption' and 'total_income' required in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if (
        "consumption" not in df_no_care_demand.columns
        or "total_income" not in df_no_care_demand.columns
    ):
        raise ValueError(
            "Columns 'consumption' and 'total_income' required in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    df_baseline["consumption_rate"] = np.where(
        df_baseline["total_income"] > 0,
        df_baseline["consumption"] / df_baseline["total_income"],
        np.nan,
    )
    df_no_care_demand["consumption_rate"] = np.where(
        df_no_care_demand["total_income"] > 0,
        df_no_care_demand["consumption"] / df_no_care_demand["total_income"],
        np.nan,
    )

    avg_cr_baseline = (
        df_baseline.groupby("age", observed=False)["consumption_rate"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_cr_baseline.columns = ["age", "avg_consumption_rate_baseline"]

    avg_cr_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["consumption_rate"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_cr_no_care_demand.columns = ["age", "avg_consumption_rate_no_care_demand"]

    merged = avg_cr_baseline.merge(avg_cr_no_care_demand, on="age", how="inner")
    merged["diff_consumption_rate"] = (
        merged["avg_consumption_rate_baseline"]
        - merged["avg_consumption_rate_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["diff_consumption_rate"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (baseline − no care demand)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Consumption rate difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_savings_dec_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "savings_dec_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot savings decision difference by age (baseline - no care demand).

    One black line.

    Period-by-period (current) savings_dec difference, not accumulated.
    Same style as other difference plots; only ages up to 89, no legend.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "savings_dec" not in df.columns:
            df["savings_dec"] = _extract_aux_variable(df, "savings_dec")

    if "savings_dec" not in df_baseline.columns:
        raise ValueError(
            "Column 'savings_dec' not found in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if "savings_dec" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'savings_dec' not found in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    avg_sd_baseline = (
        df_baseline.groupby("age", observed=False)["savings_dec"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_sd_baseline.columns = ["age", "avg_savings_dec_baseline"]

    avg_sd_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["savings_dec"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_sd_no_care_demand.columns = ["age", "avg_savings_dec_no_care_demand"]

    merged = avg_sd_baseline.merge(avg_sd_no_care_demand, on="age", how="inner")
    merged["diff_savings_dec"] = (
        merged["avg_savings_dec_baseline"] - merged["avg_savings_dec_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["diff_savings_dec"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (baseline − no care demand)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Savings dec (current) difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_consumption_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "consumption_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot consumption difference by age (baseline - no care demand).

    One black line.

    Period-by-period (current) consumption difference, not accumulated.
    Same style as other difference plots; only ages up to 89, no legend.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "consumption" not in df.columns:
            df["consumption"] = _extract_aux_variable(df, "consumption")

    if "consumption" not in df_baseline.columns:
        raise ValueError(
            "Column 'consumption' not found in baseline data. "
            f"Available: {df_baseline.columns.tolist()}"
        )
    if "consumption" not in df_no_care_demand.columns:
        raise ValueError(
            "Column 'consumption' not found in no_care_demand data. "
            f"Available: {df_no_care_demand.columns.tolist()}"
        )

    avg_cons_baseline = (
        df_baseline.groupby("age", observed=False)["consumption"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_cons_baseline.columns = ["age", "avg_consumption_baseline"]

    avg_cons_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["consumption"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_cons_no_care_demand.columns = ["age", "avg_consumption_no_care_demand"]

    merged = avg_cons_baseline.merge(avg_cons_no_care_demand, on="age", how="inner")
    merged["diff_consumption"] = (
        merged["avg_consumption_baseline"] - merged["avg_consumption_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["diff_consumption"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (baseline − no care demand)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Consumption (current) difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_accumulated_total_income_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_total_income_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot accumulated total income difference by age (baseline - no care demand).

    total_income is comprehensive household income (own + partner + transfers,
    after tax, with UB floor and child benefits).
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "total_income" not in df.columns:
            df["total_income"] = _extract_aux_variable(df, "total_income")

    avg_inc_baseline = (
        df_baseline.groupby("age", observed=False)["total_income"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_inc_baseline.columns = ["age", "avg_total_income_baseline"]

    avg_inc_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["total_income"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_inc_no_care_demand.columns = ["age", "avg_total_income_no_care_demand"]

    merged = avg_inc_baseline.merge(avg_inc_no_care_demand, on="age", how="inner")
    merged["diff_total_income"] = (
        merged["avg_total_income_baseline"] - merged["avg_total_income_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged["accumulated_diff_total_income"] = merged["diff_total_income"].cumsum()
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["accumulated_diff_total_income"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (1000 EUR)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Accumulated total income difference plot saved to {path_to_plot}")


@pytask.mark.publication
@pytask.mark.publication_post_estimation
@pytask.mark.publication_assets_and_savings
def task_plot_accumulated_own_income_difference_baseline_vs_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_baseline_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "publication"
    / "post_estimation"
    / "accumulated_own_income_difference_baseline_vs_no_care_demand.pdf",
) -> None:
    """Plot accumulated own income difference by age (baseline - no care demand).

    own_income_after_ssc = labor income + pension income (after social security
    contributions, before income tax). Isolates the direct labor-supply channel.
    """
    specs = pickle.load(path_to_specs.open("rb"))

    df_baseline = pd.read_pickle(path_to_baseline_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    for df in (df_baseline, df_no_care_demand):
        if "own_income_after_ssc" not in df.columns:
            df["own_income_after_ssc"] = _extract_aux_variable(
                df, "own_income_after_ssc"
            )

    avg_inc_baseline = (
        df_baseline.groupby("age", observed=False)["own_income_after_ssc"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_inc_baseline.columns = ["age", "avg_own_income_baseline"]

    avg_inc_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)["own_income_after_ssc"]
        .mean()
        .reset_index()
        .sort_values("age")
    )
    avg_inc_no_care_demand.columns = ["age", "avg_own_income_no_care_demand"]

    merged = avg_inc_baseline.merge(avg_inc_no_care_demand, on="age", how="inner")
    merged["diff_own_income"] = (
        merged["avg_own_income_baseline"] - merged["avg_own_income_no_care_demand"]
    )
    merged = merged.sort_values("age").reset_index(drop=True)
    merged["accumulated_diff_own_income"] = merged["diff_own_income"].cumsum()
    merged = merged[merged["age"] <= 89]

    style = PUBLICATION_PLOT_STYLE
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(10, 8))

    linewidth = 2.0
    ax.plot(
        merged["age"],
        merged["accumulated_diff_own_income"],
        color="0",
        linewidth=linewidth,
        linestyle="-",
    )

    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Difference (1000 EUR)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=8, pad=6)

    ax.grid(True, axis="y", alpha=0.10, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.14, bottom=0.12)
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    if path_to_plot.suffix.lower() == ".pdf":
        _pdf_fonttype = plt.rcParams["pdf.fonttype"]
        plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        path_to_plot,
        dpi=style["savefig_dpi"],
        bbox_inches="tight",
        pad_inches=style["savefig_pad_inches"],
    )
    if path_to_plot.suffix.lower() == ".pdf":
        plt.rcParams["pdf.fonttype"] = _pdf_fonttype
    plt.close(fig)

    print(f"Accumulated own income difference plot saved to {path_to_plot}")
