"""Plot mortality."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.utility.bequest_utility import utility_final_consume_all
from caregiving.model.utility.utility_components import consumption_scale
from caregiving.model.utility.utility_functions_additive import (
    utility_func_adda as utility_func,
)
from caregiving.specs.derive_specs import read_and_derive_specs


def task_plot_utility(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_save: Annotated[Path, Product] = BLD / "plots" / "utility" / "utility.png",
):
    """Plot utility function."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    params = yaml.safe_load(path_to_start_params.open("rb"))

    consumption = np.linspace(5_000, 100_000, 1000) / specs["wealth_unit"]
    partner_state = np.array(1)
    education = 1
    period = 35

    choice_labels = specs["choice_labels"]
    fig, ax = plt.subplots()
    for choice, choice_label in enumerate(choice_labels):
        utilities = np.zeros_like(consumption)
        for i, c in enumerate(consumption):
            utilities[i] = utility_func(
                consumption=c,
                partner_state=partner_state,
                # sex=1,
                health=1,
                care_demand=0,
                # mother_health=PARENT_DEAD,
                education=education,
                period=period,
                choice=choice,
                params=params,
                model_specs=specs,
            )
        ax.plot(
            utilities,
            consumption,
            label=choice_label,
        )
    ax.legend()
    ax.set_xlabel("Utility")
    ax.set_ylabel("Consumption")
    ax.set_title("Utility function (reversed axes)")

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.close(fig)


def task_plot_bequest(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "utility"
    / "bequest_utility.png",
):
    """Plot bequest utility."""

    specs = read_and_derive_specs(path_to_specs)

    params = yaml.safe_load(path_to_start_params.open("rb"))

    wealth = np.linspace(5_000, 100_000, 1000) / specs["wealth_unit"]

    choice_labels = specs["choice_labels"]

    fig, ax = plt.subplots()
    for _choice, choice_label in enumerate(choice_labels):
        bequests = np.zeros_like(wealth)

        for i, w in enumerate(wealth):
            bequests[i] = utility_final_consume_all(
                wealth=w,
                education=0,
                params=params,
            )
        ax.plot(
            wealth,
            bequests,
            label=choice_label,
        )

    ax.legend()
    ax.set_ylabel("Bequest Utility")
    ax.set_xlabel("Consumption")

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.close(fig)


def task_plot_cons_scale(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "utility"
    / "consumption_scale.png",
):
    """Plot conumption scale."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    n_periods = specs["n_periods"]
    married_labels = ["Single", "Partnered"]
    edu_labels = specs["education_labels"]

    fig, axs = plt.subplots(ncols=2)
    for married_val, married_label in enumerate(married_labels):

        for edu_val, edu_label in enumerate(edu_labels):
            cons_scale = np.zeros(n_periods)

            for period in range(n_periods):
                # has_partner = (np.array(married_val) > 0).astype(int)
                # n_child = specs["children_by_state"][1, edu_val, has_partner, period]
                # cons_scale[period] = consumption_scale(has_partner, n_children)
                cons_scale[period] = consumption_scale(
                    partner_state=np.array(married_val),
                    # sex=1,
                    education=edu_val,
                    period=period,
                    model_specs=specs,
                )

            axs[married_val].plot(cons_scale, label=edu_label)
            axs[married_val].set_title(married_label)
            axs[married_val].set_xlabel("Period")

            axs[married_val].legend()
            axs[married_val].set_ylim([1, 2.5])

    axs[0].set_ylabel("Consumption scale")
    fig.suptitle("Consumption scale by period")

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.close(fig)
