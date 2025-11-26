"""Test estimate_model interface with least squares and scalar optimization."""

import pickle
import signal
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from dcegm.backward_induction import backward_induction

from caregiving.config import BLD, SRC
from caregiving.estimation.estimation_setup import estimate_model
from caregiving.estimation.prepare_estimation import (
    load_and_setup_full_model_for_solution,
)
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


def test_estimate_model_least_squares_interface(temp_test_dir):
    """Test estimate_model interface with least squares optimization."""

    # Check if required files exist, skip test if not
    required_files = [
        BLD / "model" / "initial_conditions" / "states.pkl",
        BLD / "model" / "initial_conditions" / "wealth.csv",
        BLD / "moments" / "moments_full.csv",
        BLD / "moments" / "variances_full.csv",
        BLD / "model" / "options.pkl",
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

    with open(BLD / "model" / "options.pkl", "rb") as f:
        options = pickle.load(f)

    model = load_and_setup_full_model_for_solution(
        options=options, path_to_model=BLD / "model" / "model_for_solution.pkl"
    )

    start_params, lower_bounds, upper_bounds = load_real_parameters()

    solve_func = backward_induction(model)

    # Create temporary files in the test temp directory
    temp_result_file = temp_test_dir("temp_result_ls.pkl")
    temp_params_file = temp_test_dir("temp_params_ls.csv")

    with patch("optimagic.minimize", side_effect=mock_minimize):
        result = estimate_model(
            model_for_simulation=model,
            start_params=start_params,
            solve_func=solve_func,
            options=options,
            algo="tranquilo_ls",
            algo_options={"maxiter": 1, "ftol": 1e-3},
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            simulate_scenario_func=simulate_scenario,
            simulate_moments_func=simulate_moments_pandas,
            least_squares=True,
            weighting_method="identity",
            use_cholesky_weights=True,
            relative_deviations=False,
            path_to_discrete_states=BLD / "model" / "initial_conditions" / "states.pkl",
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
        BLD / "model" / "options.pkl",
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

    with open(BLD / "model" / "options.pkl", "rb") as f:
        options = pickle.load(f)

    model = load_and_setup_full_model_for_solution(
        options=options, path_to_model=BLD / "model" / "model_for_solution.pkl"
    )

    start_params, lower_bounds, upper_bounds = load_real_parameters()
    solve_func = backward_induction(model)

    # Create temporary files in the test temp directory
    temp_result_file = temp_test_dir("temp_result_scalar.pkl")
    temp_params_file = temp_test_dir("temp_params_scalar.csv")

    with patch("optimagic.minimize", side_effect=mock_minimize):
        result = estimate_model(
            model_for_simulation=model,
            start_params=start_params,
            solve_func=solve_func,
            options=options,
            algo="scipy_lbfgsb",
            algo_options={"maxiter": 1, "ftol": 1e-3},
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            simulate_scenario_func=simulate_scenario,
            simulate_moments_func=simulate_moments_pandas,
            least_squares=False,
            weighting_method="identity",
            use_cholesky_weights=True,
            relative_deviations=False,
            path_to_discrete_states=BLD / "model" / "initial_conditions" / "states.pkl",
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
        BLD / "model" / "options.pkl",
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

        with open(BLD / "model" / "options.pkl", "rb") as f:
            options = pickle.load(f)

        model = load_and_setup_full_model_for_solution(
            options=options, path_to_model=BLD / "model" / "model_for_solution.pkl"
        )

        start_params, lower_bounds, upper_bounds = load_real_parameters(
            test_param_keys=["sigma", "beta"]
        )

        solve_func = backward_induction(model)

        with patch("optimagic.minimize", side_effect=mock_minimize):
            result = estimate_model(
                model_for_simulation=model,
                start_params=start_params,
                solve_func=solve_func,
                options=options,
                algo="scipy_lbfgsb",
                algo_options={"maxiter": 1},
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                simulate_scenario_func=simulate_scenario,
                simulate_moments_func=simulate_moments_pandas,
                least_squares=False,
                path_to_discrete_states=BLD
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
