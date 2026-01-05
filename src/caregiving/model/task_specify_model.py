"""Specify model for estimation."""

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
from caregiving.model.stochastic_processes.inheritance_transition import (
    inheritance_transition,
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
from caregiving.model.wealth_and_budget.savings_grid import create_end_of_period_assets
from dcegm.pre_processing.setup_model import create_model_dict


# @pytask.mark.baseline_model
# def task_specify_model(
#     path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
#     path_to_save_model_config: Annotated[Path, Product] = BLD
#     / "model"
#     / "model_config.pkl",
#     path_to_save_model: Annotated[Path, Product] = BLD / "model" / "model.pkl",
# ):
#     model = specify_model(
#         path_to_derived_specs=path_to_derived_specs,
#         path_to_save_model_config=path_to_save_model_config,
#         path_to_save_model=path_to_save_model,
#     )

#     return model


def specify_model(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_model_config: Annotated[Path, Product] = BLD
    / "model"
    / "model_config.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD / "model" / "model.pkl",
):
    """Generate model and options dictionaries."""

    with path_to_derived_specs.open("rb") as f:
        specs = pickle.load(f)

    # Load specifications
    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=int)

    # Savings grid
    savings_grid = create_end_of_period_assets()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    model_config = {
        "min_period_batch_segments": [33, 43, 44],
        "n_periods": n_periods,
        "choices": choices,
        "deterministic_states": {
            # "partner_state": [0],
            # "health": [1],  # good health
            # "education": [0],
            # "caregiving_type": [1],
            "caregiving_type": np.arange(2, dtype=int),
            "education": np.arange(specs["n_education_types"], dtype=int),
            "already_retired": np.arange(2, dtype=int),
        },
        "stochastic_states": {
            "partner_state": np.arange(specs["n_partner_states"], dtype=int),
            "health": np.arange(specs["n_health_states"], dtype=int),
            "job_offer": np.arange(2, dtype=int),
            "mother_dead": np.arange(3, dtype=int),
            "mother_adl": np.arange(specs["n_adl_states_light_intensive"], dtype=int),
            "care_demand": np.arange(3, dtype=int),
        },
        "continuous_states": {
            "assets_end_of_period": savings_grid / specs["wealth_unit"],
            "experience": experience_grid,
        },
        "n_quad_points": specs["quadrature_points_stochastic"],
        # "n_quad_points": specs["n_quad_points"],
    }
    pickle.dump(model_config, path_to_save_model_config.open("wb"))

    # ### Now test the inspection function.
    # state_space_df = create_model_dict(
    #     model_config=model_config,
    #     model_specs=specs,
    #     state_space_functions=create_state_space_functions(),
    #     utility_functions=create_utility_functions(),
    #     utility_functions_final_period=create_final_period_utility_functions(),
    #     budget_constraint=budget_constraint,
    #     stochastic_states_transitions=create_stochastic_states_transitions(),
    #     debug_info="state_space_df",
    # )
    # admissible_df = state_space_df[state_space_df["is_valid"]]
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
        # use_stochastic_sparsity=True,
        # alternative_sim_specifications=alternative_sim_specifications,
        # debug_info="state_space_df",
    )

    print("Model specified.", flush=True)
    return model


def create_stochastic_states_transitions():
    return {
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
        "mother_adl": limitations_with_adl_transition,
        "care_demand": care_demand_transition_adl_light_intensive,
        "mother_dead": death_transition,
    }
