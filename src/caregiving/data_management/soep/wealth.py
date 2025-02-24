"""Create wealth variable from SOEP."""

import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS

# def add_wealth_interpolate_and_deflate(data, path_dict, options):
#     """Loads wealth data, interpolates linearly between first and last year of
#     observation for each household, and deflates wealth using the consumer price
#     index."""
#     data = data.reset_index()
#     wealth_data = load_wealth_data(path_dict["soep_c38"])
#     wealth_data = trim_and_rename(wealth_data)
#     wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data, options)
#     data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
#     data = deflate_wealth(data, path_dict)
#     data.loc[data["wealth"] < 0, "wealth"] = 0
#     data.set_index(["pid", "syear"], inplace=True)
#     data = data[(data["wealth"].notna())]
#     print(str(len(data)) + " left after dropping people with missing wealth.")
#     return data
