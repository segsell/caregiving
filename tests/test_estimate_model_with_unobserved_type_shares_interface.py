"""Test estimate_model_with_unobserved_type_shares interface."""

import pickle
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import optimagic as om
import pytest
import yaml

import dcegm
from caregiving.config import BLD, SRC
from caregiving.estimation.estimation_setup import (
    estimate_model_with_unobserved_type_shares,
)
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario_slim
from caregiving.simulation.simulate_moments import simulate_moments_pandas


class MockResult:
    """A pickleable mock result class for testing."""

    def __init__(
        self,
        params,
        success=True,
        message="Mock optimization completed",
        fun=0.5,
        nfev=1,
    ):
        self.params = params
        self.success = success
        self.message = message
        self.fun = fun
        self.nfev = nfev


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files and clean up after test."""
    # Create temp directory in tests/temp/
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    # List to track created files for cleanup
    created_files = []

    def create_temp_file(filename):
        """Create a temporary file in the test temp directory."""
        temp_file = temp_dir / filename
        created_files.append(temp_file)
        return temp_file

    yield create_temp_file

    # Cleanup: remove all created files
    for file_path in created_files:
        if file_path.exists():
            with suppress(OSError, PermissionError):
                file_path.unlink()


def test_estimate_model_with_unobserved_type_shares_interface(temp_test_dir):
    """Test estimate_model_with_unobserved_type_shares interface with mocked optimization."""

    # Check if required files exist, skip test if not
    required_files = [
        BLD / "model" / "initial_conditions" / "initial_states.pkl",
        BLD / "moments" / "moments_full.csv",
        BLD / "moments" / "variances_full.csv",
        BLD / "model" / "specs" / "specs_full.pkl",
        BLD / "model" / "model.pkl",
        BLD / "model" / "model_config.pkl",
        BLD / "model" / "params" / "estimated_params_model.yaml",
        SRC / "estimation" / "start_params_and_bounds" / "lower_bounds.yaml",
        SRC / "estimation" / "start_params_and_bounds" / "upper_bounds.yaml",
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        pytest.skip(f"Required files not found: {missing_files}")

    # Mock the optimization to avoid actual optimization
    def mock_minimize(**kwargs):
        """Mock optimagic minimize that returns quickly."""
        return MockResult(
            params=kwargs["params"],
            success=True,
            message="Mock optimization completed",
            fun=0.5,  # Mock objective value
            nfev=1,
        )

    # Load model specs and config
    with open(BLD / "model" / "specs" / "specs_full.pkl", "rb") as f:
        specs = pickle.load(f)

    with open(BLD / "model" / "model_config.pkl", "rb") as f:
        model_config = pickle.load(f)

    # Setup model using new interface
    model = dcegm.setup_model(
        model_specs=specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=BLD / "model" / "model.pkl",
    )

    # Load parameters from estimated_params_model.yaml
    with open(BLD / "model" / "params" / "estimated_params_model.yaml", "rb") as f:
        params = yaml.safe_load(f)

    # Load bounds
    with open(
        SRC / "estimation" / "start_params_and_bounds" / "lower_bounds.yaml"
    ) as f:
        all_lower_bounds = yaml.safe_load(f)

    with open(
        SRC / "estimation" / "start_params_and_bounds" / "upper_bounds.yaml"
    ) as f:
        all_upper_bounds = yaml.safe_load(f)

    # Extract bounds only for parameters that are in params
    lower_bounds = {
        key: all_lower_bounds[key] for key in params.keys() if key in all_lower_bounds
    }
    upper_bounds = {
        key: all_upper_bounds[key] for key in params.keys() if key in all_upper_bounds
    }

    # Create temporary files with mock_ suffix
    mock_result_file = temp_test_dir("mock_result_unobserved_type_shares.pkl")
    mock_params_file = temp_test_dir("mock_estimated_params_unobserved_type_shares.csv")

    # Mock path files - create them if they don't exist (they'll be read)
    mock_initial_states = BLD / "model" / "initial_conditions" / "initial_states.pkl"
    mock_empirical_moments = BLD / "moments" / "moments_full.csv"
    mock_empirical_variance = BLD / "moments" / "variances_full.csv"

    # Algorithm options for nag_dfols (using reasonable defaults)
    algo_options_dfols = {"maxiter": 1, "ftol": 1e-3}

    # Mock om.minimize to avoid actual optimization
    with patch("optimagic.minimize", side_effect=mock_minimize):
        # Function returns None, but saves results to files
        estimate_model_with_unobserved_type_shares(
            model_class=model,
            model_specs=specs,
            start_params=params,
            algo="nag_dfols",
            algo_options=algo_options_dfols,
            least_squares=True,
            simulate_scenario_func=simulate_scenario_slim,
            simulate_moments_func=simulate_moments_pandas,
            weighting_method="identity",
            use_cholesky_weights=True,
            relative_deviations=False,
            path_to_empirical_moments=mock_empirical_moments,
            path_to_empirical_variance=mock_empirical_variance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            scaling=om.ScalingOptions(method="bounds", clipping_value=0.1, magnitude=5),
            random_seed=False,
            path_to_save_estimation_result=mock_result_file,
            path_to_save_estimation_params=mock_params_file,
        )

    # Verify that result files were created
    assert mock_result_file.exists(), "Result file should be created"
    assert mock_params_file.exists(), "Params file should be created"

    # Verify result was saved correctly
    with open(mock_result_file, "rb") as f:
        saved_result = pickle.load(f)

    assert saved_result is not None
    assert hasattr(saved_result, "params")
    assert hasattr(saved_result, "success")

    # Verify params file was created and contains data
    assert mock_params_file.exists()
    # Verify params file contains data
    import pandas as pd

    saved_params = pd.read_csv(mock_params_file, index_col=0)
    assert len(saved_params) > 0, "Params file should contain data"

    print("✓ estimate_model_with_unobserved_type_shares interface test passed")
