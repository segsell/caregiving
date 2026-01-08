"""Set start parameters for the estimation."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import yaml
from pytask import Product, mark, task

from caregiving.config import BLD, SRC

SCENARIOS = {
    "original": {
        "path_to_start_params": SRC
        / "estimation"
        / "start_params_and_bounds"
        / "start_params.yaml",
        "path_to_save_updated_start_params": BLD
        / "model"
        / "params"
        / "start_params_updated.yaml",
    },
    "no_care_demand": {
        "path_to_start_params": SRC
        / "estimation"
        / "start_params_and_bounds"
        / "start_params_no_care_demand.yaml",
        "path_to_save_updated_start_params": BLD
        / "model"
        / "params"
        / "start_params_updated_no_care_demand.yaml",
    },
}

for scenario, scenario_params in SCENARIOS.items():

    @mark.start_params
    @task(
        name=f"task_load_and_set_start_params_{scenario}",
        kwargs={
            "path_to_job_offer_params": Path(
                BLD / "estimation" / "stochastic_processes" / "job_offer_params.csv"
            ),
            "path_to_start_params": Path(scenario_params["path_to_start_params"]),
            "path_to_save_updated_start_params": Path(
                scenario_params["path_to_save_updated_start_params"]
            ),
        },
        # produces=Path(scenario_params["path_to_save_updated_start_params"]),
    )
    def task_load_and_set_start_params(
        path_to_job_offer_params: Path,
        path_to_start_params: Path,
        path_to_save_updated_start_params: Annotated[Path, Product],
        # path_to_job_offer_params: Path = BLD
        # / "estimation"
        # / "stochastic_processes"
        # / "job_offer_params.csv",
        # path_to_save_updated_start_params: Annotated[Path, Product] = scenario_params[
        #     "path_to_save_updated_start_params"
        # ],
    ) -> None:
        """Load start parameters and update them with job offer probabilities."""
        # start_params_all = yaml.safe_load(
        #     scenario_params["path_to_start_params"].open("rb")
        # )
        start_params_all = yaml.safe_load(path_to_start_params.open("rb"))
        job_offer_params = pd.read_csv(path_to_job_offer_params, index_col=0)

        start_params_all.update(job_offer_params["value"].to_dict())

        with path_to_save_updated_start_params.open("w") as f:
            yaml.dump(start_params_all, f, default_flow_style=False)
