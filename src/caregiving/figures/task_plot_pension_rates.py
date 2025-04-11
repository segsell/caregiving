"""Plot pension rates.
source:

https://www.deutsche-rentenversicherung.de/SharedDocs/Downloads/DE/Statistiken-und-Berichte/statistikpublikationen/rv_in_zeitreihen.html

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC


def task_plot_pension_rates(
    path_to_data: Path = SRC
    / "data"
    / "statistical_office"
    / "pension_payout_and_contribution_rates.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "pension_rates.png",
):
    df_rates = pd.read_csv(path_to_data)

    years = df_rates["year"]
    replacement_rates = df_rates["replacement_rate"]
    contribution_rates = df_rates["contribution_rate"]

    fig, ax1 = plt.subplots()

    # Plot replacement rates on the left y-axis
    ax1.plot(years, replacement_rates, label="Replacement Rate", color="C0")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Replacement Rate")
    ax1.set_ylim(45, 55)
    ax1.tick_params(axis="y")

    # Create a second y-axis for the contribution rates
    ax2 = ax1.twinx()
    ax2.plot(years, contribution_rates, label="Contribution Rate", color="C1")
    ax2.set_ylabel("Contribution Rate")
    ax2.set_ylim(15, 25)
    ax2.tick_params(axis="y")

    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    plt.savefig(path_to_save)
