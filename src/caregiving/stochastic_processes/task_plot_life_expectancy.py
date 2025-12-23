"""Plot cumulative survival probability by age."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD


@pytask.mark.stochastic_processes
def task_plot_survival_by_age(
    path_to_survival_by_age: Path = BLD
    / "model"
    / "initial_conditions"
    / "survival_by_age.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "survival_by_age.png",
) -> None:
    """Plot cumulative survival probability by age.

    Parameters
    ----------
    path_to_survival_by_age : Path
        Path to CSV file with cumulative survival probabilities
    path_to_save_plot : Path
        Path to save the plot
    """
    # Load survival data
    survival_df = pd.read_csv(path_to_survival_by_age, index_col=0)
    survival_by_age = survival_df.squeeze("columns")
    survival_by_age.index = survival_by_age.index.astype(int)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        survival_by_age.index,
        survival_by_age.values,
        linewidth=2,
        color="tab:blue",
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Cumulative Survival Probability")
    ax.set_title("Cumulative Survival Probability by Age")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(survival_by_age.index.min(), survival_by_age.index.max())
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)
