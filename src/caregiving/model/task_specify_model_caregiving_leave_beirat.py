"""Specify model for Beirat caregiving leave (max 3 years, max 1 full leave).

Implements the Pflegebeirat recommendation: total leave cap 3 years, of which
at most 1 can be full leave; up to 3 years partial leave if no full leave is
taken. Baseline Pflegegeld (care cash benefits) kept in place alongside the
65% leave top-up.
"""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pytask
from pytask import Product

import dcegm
from caregiving.config import BLD
from caregiving.model.experience_baseline_model import define_experience_grid
from caregiving.model.state_space_caregiving_leave_beirat import (
    create_state_space_functions,
)
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
from caregiving.model.stochastic_processes.job_transition_job_retention import (
    job_offer_process_transition_beirat,
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
from caregiving.model.wealth_and_budget.budget_equation_caregiving_leave_beirat import (  # noqa: E501
    budget_constraint,
)
from caregiving.model.wealth_and_budget.savings_grid import create_end_of_period_assets


@pytask.mark.caregiving_leave_beirat_model
def task_specify_model_caregiving_leave_beirat(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_model_config: Annotated[Path, Product] = BLD
    / "model"
    / "model_config_caregiving_leave_beirat.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD
    / "model"
    / "model_caregiving_leave_beirat.pkl",
):
    """Build Beirat caregiving leave model (3-year cap, 1 full leave, Pflegegeld kept)."""
    model = specify_model_caregiving_leave_beirat(
        path_to_derived_specs=path_to_derived_specs,
        path_to_save_model_config=path_to_save_model_config,
        path_to_save_model=path_to_save_model,
    )
    return model


def specify_model_caregiving_leave_beirat(
    path_to_derived_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_model_config: Annotated[Path, Product] = BLD
    / "model"
    / "model_config_caregiving_leave_beirat.pkl",
    path_to_save_model: Annotated[Path, Product] = BLD
    / "model"
    / "model_caregiving_leave_beirat.pkl",
):
    """Generate model for Beirat caregiving leave counterfactual.

    - job_before_caregiving (0/1/2), years_leave_used_total (0..3), full_leave_year_used (0/1)
    - Job retention and 65% leave top-up only when years_leave_used_total < 3
    - Full-leave benefit only when full_leave_year_used == 0
    - Baseline Pflegegeld (calc_care_benefits_and_costs) included in budget
    """
    with path_to_derived_specs.open("rb") as f:
        specs = pickle.load(f)

    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=int)

    savings_grid = create_end_of_period_assets()
    experience_grid = define_experience_grid(specs)

    model_config = {
        "min_period_batch_segments": [33, 43, 44],
        "n_periods": n_periods,
        "choices": choices,
        "deterministic_states": {
            "caregiving_type": np.arange(2, dtype=int),
            "education": np.arange(specs["n_education_types"], dtype=int),
            "already_retired": np.arange(2, dtype=int),
            "job_before_caregiving": np.arange(3, dtype=int),  # 0=none, 1=PT, 2=FT
            "years_leave_used_total": np.arange(4, dtype=int),  # 0..3
            "full_leave_year_used": np.arange(2, dtype=int),   # 0 or 1
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

    print("Beirat caregiving leave model specified.", flush=True)

    return model


def create_stochastic_states_transitions():
    """Stochastic state transitions for Beirat model."""
    return {
        "job_offer": job_offer_process_transition_beirat,
        "partner_state": partner_transition,
        "health": health_transition,
        "mother_dead": death_transition,
        "mother_adl": limitations_with_adl_transition,
        "care_demand": care_demand_transition_adl_light_intensive,
    }
