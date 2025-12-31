"""Specify counterfactual model without care_demand process."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pytask
import yaml
from pytask import Product

import dcegm
from caregiving.config import BLD

from caregiving.model.state_space_no_care_demand import create_state_space_functions
from caregiving.model.stochastic_processes.adl_transition import death_transition
from caregiving.model.stochastic_processes.health_transition import (
    health_transition,
)
from caregiving.model.stochastic_processes.job_transition_no_care_demand import (
    job_offer_process_transition,
)
from caregiving.model.stochastic_processes.partner_transition import partner_transition
from caregiving.model.taste_shocks import shock_function_dict
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


@pytask.mark.no_care_demand_model
def task_specify_model_no_care_demand(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_model_config: Annotated[Path, Product] = BLD
    / "model"
    / "model_config_no_care_demand.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD
    / "model"
    / "model_no_care_demand.pkl",
):
    """Generate counterfactual model without care_demand, caregiving_type, mother_adl.

    This counterfactual removes most care-related processes from the state space:
    - care_demand
    - caregiving_type
    - mother_adl

    Note: mother_dead is kept to allow inheritance calculation (even though there's no care).

    All other specifications are kept identical to the baseline.
    """

    with path_to_derived_specs.open("rb") as f:
        specs = pickle.load(f)

    # Load specifications
    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=int)

    # Savings grid
    savings_grid = create_savings_grid()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    # Build model_config without care-related states
    model_config = {
        "min_period_batch_segments": [33, 43, 44],
        "n_periods": n_periods,
        "choices": choices,
        "deterministic_states": {
            "caregiving_type": np.arange(2, dtype=int),
            "education": np.arange(specs["n_education_types"], dtype=int),
            "already_retired": np.arange(2, dtype=int),
        },
        "stochastic_states": {
            "partner_state": np.arange(specs["n_partner_states"], dtype=int),
            "health": np.arange(specs["n_health_states"], dtype=int),
            "job_offer": np.arange(2, dtype=int),
            "mother_dead": np.arange(
                3, dtype=int
            ),  # Needed for inheritance calculation
            # No mother_adl or care_demand in counterfactual
        },
        "continuous_states": {
            "assets_end_of_period": savings_grid,
            "experience": experience_grid,
        },
        "n_quad_points": specs["quadrature_points_stochastic"],
    }
    pickle.dump(model_config, path_to_save_model_config.open("wb"))

    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_save_path=path_to_save_model,
    )
    print("Counterfactual model without care_demand specified.", flush=True)

    return model


def create_stochastic_states_transitions():
    """Create stochastic state transitions for no care demand counterfactual.

    Excludes care_demand and mother_adl transitions, but includes mother_dead for inheritance.

    """
    return {
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
        "mother_dead": death_transition,  # Needed for inheritance calculation
        # No mother_adl or care_demand in counterfactual
    }
