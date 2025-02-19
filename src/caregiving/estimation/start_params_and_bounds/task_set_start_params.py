"""Set start parameters for the estimation."""

import pickle as pkl
from pathlib import Path
from typing import Annotated, Any, Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from pytask import Product

from caregiving.config import BLD, SRC


def task_load_and_set_start_params(
    path_to_start_params: Path = SRC
    / "estimation"
    / "start_params_and_bounds"
    / "start_params.yaml",
    path_to_job_offer_params: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "job_offer_params.csv",
    path_to_save_updated_start_params: Annotated[Path, Product] = BLD
    / "model"
    / "start_params_and_bounds"
    / "start_params_updated.yaml",
) -> None:
    """Load start parameters and update them with job offer probabilities."""
    start_params_all = yaml.safe_load(path_to_start_params.open("rb"))

    job_offer_params = pd.read_csv(path_to_job_offer_params, index_col=0)

    start_params_all.update(job_offer_params["value"].to_dict())

    with path_to_save_updated_start_params.open("w") as f:
        yaml.dump(start_params_all, f, default_flow_style=False)
