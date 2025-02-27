"""Shared utilities."""

import pickle
from pathlib import Path

import pandas as pd


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


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
