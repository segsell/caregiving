# from pathlib import Path
# from typing import Annotated, Any

# import pytask
# from dcegm.pre_processing.setup_model import setup_and_save_model
# from pytask import Product

# from caregiving.config import BLD, SRC
# from caregiving.counterfactual.state_space_counterfactual import (
#     create_state_space_functions_no_informal_care,
#     create_state_space_functions_only_informal_care,
# )
# from caregiving.model.budget import budget_constraint
# from caregiving.model.task_specify_model import get_options_dict
# from caregiving.model.utility_functions import (
#     create_final_period_utility_functions,
#     create_utility_functions,
# )


# @pytask.mark.skip(reason="Respecifying model.")
# def task_specify_and_setup_model_only_informal_care(
#     path_to_specs: Path = SRC / "model" / "specs.yaml",
#     path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "model"
#     / "counterfactual_only_informal_care.pkl",
# ) -> dict[str, Any]:
#     """Generate options and setup model.

#     start_params["sigma"] = specs["income_shock_scale"]

#     """
#     options = get_options_dict(path_to_specs, path_to_exog)

#     return setup_and_save_model(
#         options=options,
#         state_space_functions=create_state_space_functions_only_informal_care(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_save,
#     )


# @pytask.mark.skip(reason="Respecifying model.")
# def task_specify_and_setup_model_no_informal_care(
#     path_to_specs: Path = SRC / "model" / "specs.yaml",
#     path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
#     path_to_save: Annotated[Path, Product] = BLD
#     / "model"
#     / "counterfactual_no_informal_care.pkl",
# ) -> dict[str, Any]:
#     """Generate options and setup model.

#     start_params["sigma"] = specs["income_shock_scale"]

#     """
#     options = get_options_dict(path_to_specs, path_to_exog)

#     return setup_and_save_model(
#         options=options,
#         state_space_functions=create_state_space_functions_no_informal_care(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         path=path_to_save,
#     )
