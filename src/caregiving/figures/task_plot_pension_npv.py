"""Plot Net Present Value of Pensions."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.model.wealth_and_budget.tax_and_ssc import (
    calc_after_ssc_income_pensioneer,
)


def task_plot_pension_npv_by_age(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "pension_npv.png",
):
    """Plot Net Present Value of Pensions."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Select highest pension point value
    pension_point_value = specs["monthly_pension_point_value_west_2010"]
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    discount_rate = 0.03

    # calculate net periodic retirement income (assumption: working until 67)
    pension_factor = 1
    experience = 67 - start_age
    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    retirement_income_net = calc_after_ssc_income_pensioneer(retirement_income_gross)

    # calculate net present value of retirement income at age 67
    npv_67 = retirement_income_net / discount_rate - (
        retirement_income_net / discount_rate
    ) / (1 + discount_rate) ** (end_age - 67)

    # calculate net present value of retirement income at different ages
    npv_by_age = np.full(67 - start_age + 1, npv_67)
    discount_factor_by_age = np.power(1 + discount_rate, np.arange(67 - start_age + 1))
    npv_by_age = npv_by_age / discount_factor_by_age
    npv_by_age_reversed = npv_by_age[::-1]
    plt.plot(np.arange(start_age, 68), npv_by_age_reversed)
    plt.xlabel("Age")
    plt.ylabel("NPV of retirement income (1000 €)")
