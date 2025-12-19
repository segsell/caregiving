"""Specify model for estimation."""

import pickle
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import numpy as np
import pytask
import yaml

# from dcegm.pre_processing.setup_model import setup_and_save_model
from pytask import Product

import dcegm
from caregiving.config import BLD
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.stochastic_processes.adl_transition import (
    death_transition,
    limitations_with_adl_transition,
)
from caregiving.model.stochastic_processes.caregiving_transition import (
    care_demand_transition_adl_light_intensive,
)
from caregiving.model.stochastic_processes.health_transition import (
    health_transition,
)
from caregiving.model.stochastic_processes.job_transition import (
    job_offer_process_transition,
)
from caregiving.model.stochastic_processes.partner_transition import (
    partner_transition,
)
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import (
    create_utility_functions,
)
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.savings_grid import create_savings_grid


@pytask.mark.baseline_model
def task_specify_model(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_start_params: Path = (
        BLD / "model" / "params" / "start_params_updated.yaml"
    ),
    path_to_save_options: Annotated[Path, Product] = BLD / "model" / "options.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD / "model" / "model_config.pkl",
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

    # Load specifications
    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=int)

    # Savings grid
    savings_grid = create_savings_grid()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    model_config = {
        # "min_period_batch_segments": [33 - 5, 44 - 5],
        # "min_period_batch_segments": [44 - 5],
        "n_periods": n_periods,
        "choices": choices,
        "deterministic_states": {
            # "partner_state": [0],
            # "education": [0],
            # "caregiving_type": [0],
            "caregiving_type": np.arange(2, dtype=int),
            "education": np.arange(specs["n_education_types"], dtype=int),
            "already_retired": np.arange(2, dtype=int),
        },
        "stochastic_states": {
            "job_offer": np.arange(2, dtype=int),
            "partner_state": np.arange(specs["n_partner_states"], dtype=int),
            "health": np.arange(specs["n_health_states"], dtype=int),
            "mother_dead": np.arange(2, dtype=int),
            "mother_adl": np.arange(specs["n_adl_states_light_intensive"], dtype=int),
            "care_demand": np.arange(3, dtype=int),
        },
        "continuous_states": {
            "assets_end_of_period": savings_grid,
            "experience": experience_grid,
        },
        "n_quad_points": specs["quadrature_points_stochastic"],
        # "n_quad_points": specs["n_quad_points"],
    }

    stochastic_states_transitions = {
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
        "mother_dead": death_transition,
        "mother_adl": limitations_with_adl_transition,
        "care_demand": care_demand_transition_adl_light_intensive,
    }

    # model = setup_and_save_model(
    #     options=options,
    #     state_space_functions=create_state_space_functions(),
    #     utility_functions=create_utility_functions(),
    #     utility_functions_final_period=create_final_period_utility_functions(),
    #     budget_constraint=budget_constraint,
    #     # shock_functions=shock_function_dict(),
    #     path=path_to_save_model,
    #     sim_model=False,
    # )

    model = dcegm.setup_model(
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
        # debug_info=None,
        # use_stochastic_sparsity=True,
    )

    print("Model specified.", flush=True)
    return model, params
