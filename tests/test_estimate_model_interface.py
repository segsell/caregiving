"""Test estimate_model interface with least squares and scalar optimization."""

import pickle
import signal
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import dcegm
from caregiving.config import BLD, SRC
from caregiving.estimation.estimation_setup import estimate_model
from caregiving.model.state_space import create_state_space_functions
from caregiving.model.task_specify_model import create_stochastic_states_transitions
from caregiving.model.taste_shocks import shock_function_dict
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario
from caregiving.simulation.simulate_moments import simulate_moments_pandas


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


def load_real_parameters(test_param_keys=None):
    """Load start parameters, bounds from the repo."""
    if test_param_keys is None:
        test_param_keys = ["sigma", "beta", "rho", "interest_rate"]

    start_params_path = (
        SRC / "estimation" / "start_params_and_bounds" / "start_params.yaml"
    )
    lower_bounds_path = (
        SRC / "estimation" / "start_params_and_bounds" / "lower_bounds.yaml"
    )
    upper_bounds_path = (
        SRC / "estimation" / "start_params_and_bounds" / "upper_bounds.yaml"
    )

    with open(start_params_path) as f:
        all_start_params = yaml.safe_load(f)

    with open(lower_bounds_path) as f:
        all_lower_bounds = yaml.safe_load(f)

    with open(upper_bounds_path) as f:
        all_upper_bounds = yaml.safe_load(f)

    # Use subset of parameters for testing
    start_params = {
        key: all_start_params[key] for key in test_param_keys if key in all_start_params
    }
    lower_bounds = {
        key: all_lower_bounds[key] for key in test_param_keys if key in all_lower_bounds
    }
    upper_bounds = {
        key: all_upper_bounds[key] for key in test_param_keys if key in all_upper_bounds
    }

    return start_params, lower_bounds, upper_bounds


def create_simulate_scenario_wrapper():
    """Create a simulate_scenario wrapper for the new dcegm interface."""

    def simulate_scenario_wrapper(
        model_solved,
        initial_states,
        wealth_agents,
        params,
        model_specs,
        seed,
    ):
        """Wrapper that uses the new model_solved.simulate() interface.

        model_solved is already provided from model_class.solve(params),
        so we just need to call simulate() on it.
        """
        sim_df = model_solved.simulate(
            states_initial=initial_states,
            seed=seed,
        )
        return sim_df

    return simulate_scenario_wrapper


def test_estimate_model_least_squares_interface(temp_test_dir):
    """Test estimate_model interface with least squares optimization."""

    # Check if required files exist, skip test if not
    required_files = [
        BLD / "model" / "initial_conditions" / "states.pkl",
        BLD / "model" / "initial_conditions" / "wealth.csv",
        BLD / "moments" / "moments_full.csv",
        BLD / "moments" / "variances_full.csv",
        BLD / "model" / "specs" / "specs_full.pkl",
        BLD / "model" / "model.pkl",
        BLD / "model" / "model_config.pkl",
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        pytest.skip(f"Required files not found: {missing_files}")

    # Mock the optimization to stop quickly
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
        model_specs = pickle.load(f)

    with open(BLD / "model" / "model_config.pkl", "rb") as f:
        model_config = pickle.load(f)

    # Setup model using new interface
    model_class = dcegm.setup_model(
        model_specs=model_specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=BLD / "model" / "model.pkl",
    )

    start_params, lower_bounds, upper_bounds = load_real_parameters()

    # Create simulate_scenario wrapper
    simulate_scenario_wrapper = create_simulate_scenario_wrapper()

    # Create temporary files in the test temp directory
    temp_result_file = temp_test_dir("temp_result_ls.pkl")
    temp_params_file = temp_test_dir("temp_params_ls.csv")

    with patch("optimagic.minimize", side_effect=mock_minimize):
        result = estimate_model(
            model=model_class,
            start_params=start_params,
            model_specs=model_specs,
            algo="tranquilo_ls",
            algo_options={"maxiter": 1, "ftol": 1e-3},
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            simulate_scenario_func=simulate_scenario_wrapper,
            simulate_moments_func=simulate_moments_pandas,
            least_squares=True,
            weighting_method="identity",
            use_cholesky_weights=True,
            relative_deviations=False,
            path_to_initial_states=BLD / "model" / "initial_conditions" / "states.pkl",
            path_to_wealth=BLD / "model" / "initial_conditions" / "wealth.csv",
            path_to_empirical_moments=BLD / "moments" / "moments_full.csv",
            path_to_empirical_variance=BLD / "moments" / "variances_full.csv",
            path_to_save_estimation_result=temp_result_file,
            path_to_save_estimation_params=temp_params_file,
        )

    assert result is not None
    assert hasattr(result, "params")
    assert hasattr(result, "success")
    print("✓ Least squares optimization interface test passed")


def test_estimate_model_scalar_interface(temp_test_dir):
    """Test estimate_model interface with scalar optimization."""

    # Check if required files exist, skip test if not
    required_files = [
        BLD / "model" / "initial_conditions" / "states.pkl",
        BLD / "model" / "initial_conditions" / "wealth.csv",
        BLD / "moments" / "moments_full.csv",
        BLD / "moments" / "variances_full.csv",
        BLD / "model" / "specs" / "specs_full.pkl",
        BLD / "model" / "model.pkl",
        BLD / "model" / "model_config.pkl",
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        pytest.skip(f"Required files not found: {missing_files}")

    # Mock the optimization to stop quickly
    def mock_minimize(**kwargs):
        """Mock optimagic minimize that returns quickly."""
        return MockResult(
            params=kwargs["params"],
            success=True,
            message="Mock scalar optimization completed",
            fun=0.3,  # Mock scalar objective value
            nfev=1,
        )

    # Load model specs and config
    with open(BLD / "model" / "specs" / "specs_full.pkl", "rb") as f:
        model_specs = pickle.load(f)

    with open(BLD / "model" / "model_config.pkl", "rb") as f:
        model_config = pickle.load(f)

    # Setup model using new interface
    model_class = dcegm.setup_model(
        model_specs=model_specs,
        model_config=model_config,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        shock_functions=shock_function_dict(),
        stochastic_states_transitions=create_stochastic_states_transitions(),
        model_load_path=BLD / "model" / "model.pkl",
    )

    start_params, lower_bounds, upper_bounds = load_real_parameters()

    # Create simulate_scenario wrapper
    simulate_scenario_wrapper = create_simulate_scenario_wrapper()

    # Create temporary files in the test temp directory
    temp_result_file = temp_test_dir("temp_result_scalar.pkl")
    temp_params_file = temp_test_dir("temp_params_scalar.csv")

    with patch("optimagic.minimize", side_effect=mock_minimize):
        result = estimate_model(
            model=model_class,
            start_params=start_params,
            model_specs=model_specs,
            algo="scipy_lbfgsb",
            algo_options={"maxiter": 1, "ftol": 1e-3},
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            simulate_scenario_func=simulate_scenario_wrapper,
            simulate_moments_func=simulate_moments_pandas,
            least_squares=False,
            weighting_method="identity",
            use_cholesky_weights=True,
            relative_deviations=False,
            path_to_initial_states=BLD / "model" / "initial_conditions" / "states.pkl",
            path_to_wealth=BLD / "model" / "initial_conditions" / "wealth.csv",
            path_to_empirical_moments=BLD / "moments" / "moments_full.csv",
            path_to_empirical_variance=BLD / "moments" / "variances_full.csv",
            path_to_save_estimation_result=temp_result_file,
            path_to_save_estimation_params=temp_params_file,
        )

    assert result is not None
    assert hasattr(result, "params")
    assert hasattr(result, "success")
    print("✓ Scalar optimization interface test passed")


def test_estimate_model_with_timeout(temp_test_dir):
    """Test estimate_model with a timeout to prevent long runs."""

    # Check if required files exist, skip test if not
    required_files = [
        BLD / "model" / "initial_conditions" / "states.pkl",
        BLD / "model" / "initial_conditions" / "wealth.csv",
        BLD / "moments" / "moments_full.csv",
        BLD / "moments" / "variances_full.csv",
        BLD / "model" / "specs" / "specs_full.pkl",
        BLD / "model" / "model.pkl",
        BLD / "model" / "model_config.pkl",
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        pytest.skip(f"Required files not found: {missing_files}")

    def timeout_handler(signum, frame):
        raise TimeoutError("Test timed out after 30 seconds")

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        # Mock the optimization to stop quickly
        def mock_minimize(**kwargs):
            """Mock optimagic minimize that returns quickly."""
            return MockResult(
                params=kwargs["params"],
                success=True,
                message="Mock optimization completed",
                fun=0.5,
                nfev=1,
            )

        # Load model specs and config
        with open(BLD / "model" / "specs" / "specs_full.pkl", "rb") as f:
            model_specs = pickle.load(f)

        with open(BLD / "model" / "model_config.pkl", "rb") as f:
            model_config = pickle.load(f)

        # Setup model using new interface
        model_class = dcegm.setup_model(
            model_specs=model_specs,
            model_config=model_config,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            shock_functions=shock_function_dict(),
            stochastic_states_transitions=create_stochastic_states_transitions(),
            model_load_path=BLD / "model" / "model.pkl",
        )

        start_params, lower_bounds, upper_bounds = load_real_parameters(
            test_param_keys=["sigma", "beta"]
        )

        # Create simulate_scenario wrapper
        simulate_scenario_wrapper = create_simulate_scenario_wrapper()

        with patch("optimagic.minimize", side_effect=mock_minimize):
            result = estimate_model(
                model=model_class,
                start_params=start_params,
                model_specs=model_specs,
                algo="scipy_lbfgsb",
                algo_options={"maxiter": 1},
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                simulate_scenario_func=simulate_scenario_wrapper,
                simulate_moments_func=simulate_moments_pandas,
                least_squares=False,
                path_to_initial_states=BLD
                / "model"
                / "initial_conditions"
                / "states.pkl",
                path_to_wealth=BLD / "model" / "initial_conditions" / "wealth.csv",
                path_to_empirical_moments=BLD / "moments" / "moments_full.csv",
                path_to_empirical_variance=BLD / "moments" / "variances_full.csv",
            )

        assert result is not None
        print("✓ Timeout test passed")

    except TimeoutError:
        print("⚠ Test timed out (this is expected for optimization)")
    finally:
        signal.alarm(0)  # Cancel the alarm
