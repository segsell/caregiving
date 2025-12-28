"""Shared utilities."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def table(df_col):
    """Return frequency table."""
    if isinstance(df_col, np.ndarray):
        return pd.Series(df_col).value_counts().sort_index()
    else:
        return pd.crosstab(df_col, columns="Count")["Count"]


def describe(df_col):
    """Return descriptive statistics."""
    return df_col.describe()


def count(df_col):
    """Count the number of non-missing observations."""
    return df_col.count()


def statsmodels_params_to_dict(params, name_prefix, name_constant=None):
    """Turn statsmodels regression params object into dict.

    Args:
        params (pd.Series): Pandas Series containing the parameter names and values.
        name_constant (str): A custom string to use in the new name for 'const'.
        name_prefix (str): A custom prefix to prepend to all parameter names.

    Returns:
        dict: A dictionary with regression parameters.

    """
    name_constant = "" if name_constant is None else name_constant + "_"

    return {
        f"{name_prefix}_{(f'{name_constant}constant' if key == 'const' else key)}": val
        for key, val in params.items()
    }


def save_dict_to_pickle(data_dict, file_path):
    """Saves a Python dictionary to a pickle file.

    Args:
        data_dict (dict): The dictionary to be saved.
        file_path (str): The path of the file where the dictionary will be saved.

    Returns:
        None

    """
    with Path.open(file_path, "wb") as file:
        pickle.dump(data_dict, file)


def load_dict_from_pickle(file_path):
    """Loads a Python dictionary from a pickle file.

    Args:
        file_path (str): The path of the pickle file from which to load the dictionary.

    Returns:
        dict: The loaded dictionary.

    """
    with Path.open(file_path, "rb") as file:
        return pickle.load(file)


def create_age_bins(start_age, end_age, bin_size, min_remainder_size=2):
    """Create age bins with flexible bin size and optional smaller final bin.

    Args:
        start_age (int): Starting age (inclusive).
        end_age (int): Ending age (inclusive).
        bin_size (int): Preferred bin size (e.g., 3 for 3-year bins, 5 for 5-year bins).
        min_remainder_size (int): Minimum size for a remainder bin. If the remainder is
            smaller than bin_size but >= min_remainder_size, a smaller bin is created.
            Default is 2.

    Returns:
        tuple or list:
            - If jax_format=False: tuple of (bin_edges, bin_labels) where:
                - bin_edges (list): List of bin edges suitable for pd.cut
                - bin_labels (list): List of bin labels in format "start_end"
            - If jax_format=True: list of (start, end) tuples

    Examples:
        >>> create_age_bins(45, 68, 3, 2)  # doctest: +NORMALIZE_WHITESPACE
        ([45, 48, 51, 54, 57, 60, 63, 66, 69],
         ['45_47', '48_50', '51_53', '54_56', '57_59',
          '60_62', '63_65', '66_68'])

        >>> create_age_bins(45, 69, 5, 2)  # doctest: +NORMALIZE_WHITESPACE
        ([45, 50, 55, 60, 65, 70],
         ['45_49', '50_54', '55_59', '60_64', '65_69'])

    """
    bin_edges = []
    bin_labels = []
    start = start_age

    # Create full bins of the specified size
    while start <= end_age - (bin_size - 1):
        end = start + bin_size
        bin_edges.append(start)
        bin_labels.append(f"{start}_{end - 1}")
        start = end

    # Handle remainder
    if start <= end_age:
        # Remainder exists
        remainder_size = end_age - start + 1
        if remainder_size >= min_remainder_size:
            # Create a smaller final bin
            bin_edges.append(start)
            bin_labels.append(f"{start}_{end_age}")
            bin_edges.append(end_age + 1)  # Right edge
        else:
            # Remainder too small, just add right edge
            bin_edges.append(start)
    else:
        # No remainder, just add right edge for the last full bin
        bin_edges.append(start)

    return (bin_edges, bin_labels)
