"""Specify counterfactual model without care_demand process."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import yaml
from dcegm.pre_processing.setup_model import setup_and_save_model
from pytask import Product

from caregiving.config import BLD
from caregiving.model.state_space_no_care_demand import create_state_space_functions
from caregiving.model.stochastic_processes.health_transition import (
    health_transition,
)
from caregiving.model.stochastic_processes.job_transition_no_care_demand import (
    job_offer_process_transition,
)
from caregiving.model.stochastic_processes.partner_transition import partner_transition
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive_no_care_demand import (
    create_utility_functions,
)
from caregiving.model.wealth_and_budget.budget_equation_no_care_demand import (
    budget_constraint,
)
from caregiving.model.wealth_and_budget.savings_grid import create_savings_grid


def task_specify_model_no_care_demand(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_start_params: Path = BLD
    / "model"
    / "params"
    / "start_params_updated_no_care_demand.yaml",
    path_to_save_options: Annotated[Path, Product] = BLD
    / "model"
    / "options_no_care_demand.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD
    / "model"
    / "model_no_care_demand.pkl",
    path_to_save_start_params: Annotated[Path, Product] = BLD
    / "model"
    / "params"
    / "start_params_model_no_care_demand.yaml",
):
    """Generate counterfactual model and options dictionaries without care_demand.

    This counterfactual removes the ``care_demand`` exogenous process from the
    state space. All other specifications are kept identical to the baseline.
    """

    with path_to_derived_specs.open("rb") as f:
        specs = pickle.load(f)

    params = yaml.safe_load(path_to_start_params.open("rb"))

    # Assign selected specs to params written alongside the model
    params["sigma"] = float(specs["income_shock_scale"])  # income shock scale
    params["interest_rate"] = float(specs["interest_rate"])  # interest rate
    params["beta"] = float(specs["discount_factor"])  # discount factor

    with path_to_save_start_params.open("w") as f:
        yaml.dump(params, f)

    # Load specifications
    n_periods = specs["n_periods"]
    choices = np.arange(4, dtype=int)

    # Savings grid
    savings_grid = create_savings_grid()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    # Build options with exogenous processes EXCLUDING "care_demand"
    options = {
        "state_space": {
            "min_period_batch_segments": [33, 44],
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "education": np.arange(specs["n_education_types"], dtype=int),
                "already_retired": np.arange(2, dtype=int),
                "has_sister": np.arange(2, dtype=int),
            },
            "exogenous_processes": {
                "job_offer": {
                    "transition": job_offer_process_transition,
                    "states": np.arange(2, dtype=int),
                },
                "partner_state": {
                    "transition": partner_transition,
                    "states": np.arange(specs["n_partner_states"], dtype=int),
                },
                "health": {
                    "transition": health_transition,
                    "states": np.arange(specs["n_health_states"], dtype=int),
                },
            },
            "continuous_states": {
                "wealth": savings_grid,
                "experience": experience_grid,
            },
        },
        "model_params": specs,
    }

    pickle.dump(options, path_to_save_options.open("wb"))

    model = setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_save_model,
        sim_model=False,
    )

    print("Counterfactual model without care_demand specified.")
    return model, params
