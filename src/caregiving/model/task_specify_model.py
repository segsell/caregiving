"""Specify model for estimation."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import yaml
from dcegm import setup_model
from pytask import Product

from caregiving.config import BLD
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_and_supply_transition,
    health_transition_good_medium_bad,
)
from caregiving.model.stochastic_processes.health_transition import (
    health_transition,
)
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition,
)
from caregiving.model.stochastic_processes.partner_transition import partner_transition
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.savings_grid import create_savings_grid
from caregiving.model.taste_shocks import shock_function_dict


def task_specify_model(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_start_params: Path = (
        BLD / "model" / "params" / "start_params_updated.yaml"
    ),
    # path_to_save_options: Annotated[Path, Product] = BLD / "model" / "options.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD
    / "model"
    / "model_for_solution.pkl",
    path_to_save_start_params: Annotated[Path, Product] = BLD
    / "model"
    / "params"
    / "start_params_model.yaml",
):
    """Generate model and options dictionaries."""

    with path_to_derived_specs.open("rb") as f:
        specs = pickle.load(f)

    params = yaml.safe_load(path_to_start_params.open("rb"))

    # Assign income shock scale to start_params_all
    params["sigma"] = float(specs["income_shock_scale"])
    params["interest_rate"] = float(specs["interest_rate"])
    params["beta"] = float(specs["discount_factor"])

    with path_to_save_start_params.open("w") as f:
        yaml.dump(params, f)

    # Savings grid
    savings_grid = create_savings_grid()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    # Stochastic transition functions
    stochastic_states_transitions = {
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
        "mother_health": health_transition_good_medium_bad,
        "care_demand": care_demand_and_supply_transition,
    }

    model_config = {
        "state_space": {
            "min_period_batch_segments": [33, 43, 44],
            "n_periods": specs["n_periods"],
            "choices": np.arange(specs["n_choices"], dtype=int),
            "deterministic_states": {
                "education": np.arange(specs["n_education_types"], dtype=int),
                "has_sister": np.arange(2, dtype=int),
                "already_retired": np.arange(2, dtype=int),
            },
            "stochastic_states": {
                "job_offer": np.arange(2, dtype=int),
                "partner_state": np.arange(specs["n_partner_states"], dtype=int),
                "health": np.arange(specs["n_health_states"], dtype=int),
                "mother_health": np.arange(specs["n_health_states_three"], dtype=int),
                "care_demand": np.arange(3, dtype=int),
            },
            "continuous_states": {
                "asssets_end_of_period": savings_grid,
                "experience": experience_grid,
            },
            "n_quad_points": specs["n_quad_points"],
        },
    }
    # pickle.dump(model_config, path_to_save_options.open("wb"))

    model = setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=stochastic_states_transitions,
        model_save_path=path_to_save_model,
        # alternative_sim_specifications=alternative_sim_specifications,
        # debug_info=debug_info,
        use_stochastic_sparsity=False,
    )

    print("Model specified.", flush=True)
    return model
