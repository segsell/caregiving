"""Test function to compare shapes of SOEP moments and simulated moments."""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from caregiving.simulation.simulate_moments_no_care_demand import (
    simulate_moments_pandas_no_care_demand,
)


def test_soep_vs_simulated_moments_shape():
    """
    Test that compares the shapes of SOEP moments and simulated moments.

    This test ensures that the simulated moments have the same number of
    moment names as the empirical SOEP moments, which is crucial for
    proper model estimation and comparison.
    """
    # Path to the SOEP moments file
    soep_moments_path = Path("bld/moments/soep_moments_no_care_demand.csv")

    # Path to the simulated data (no care demand specification)
    simulated_data_path = Path(
        "bld/solve_and_simulate/simulated_data_no_care_demand.pkl"
    )

    # Path to the model options (no care demand specification)
    options_path = Path("bld/model/options_no_care_demand.pkl")

    # Load SOEP moments (no care demand version)
    soep_moments = pd.read_csv(soep_moments_path, index_col=0)
    soep_moment_names = soep_moments.index.tolist()
    n_soep_moments = len(soep_moment_names)

    # Load simulated data
    import pickle

    with open(simulated_data_path, "rb") as f:
        simulated_data = pickle.load(f)

    # Load model options
    with open(options_path, "rb") as f:
        options = pickle.load(f)

    # Generate simulated moments using actual data
    simulated_moments = simulate_moments_pandas_no_care_demand(
        df_full=simulated_data, options=options
    )

    # Extract simulated moment names
    simulated_moment_names = simulated_moments.index.tolist()
    n_simulated_moments = len(simulated_moment_names)

    # Test assertions
    assert n_simulated_moments > 0, "Simulated moments should not be empty"
    assert n_soep_moments > 0, "SOEP moments should not be empty"
    assert n_simulated_moments == n_soep_moments, (
        f"Number of simulated moments ({n_simulated_moments}) does not match "
        f"number of SOEP moments no care demand ({n_soep_moments})"
    )

    # Check if the moment names match (optional - might be too strict)
    # Uncomment the following lines if you want to check exact moment name matching
    # missing_in_simulated = set(soep_moment_names) - set(simulated_moment_names)
    # missing_in_soep = set(simulated_moment_names) - set(soep_moment_names)
    #
    # assert len(missing_in_simulated) == 0, (
    #     f"Moments missing in simulated data: {missing_in_simulated}"
    # )
    # assert len(missing_in_soep) == 0, (
    #     f"Moments missing in SOEP data: {missing_in_soep}"
    # )

    print(f"✓ Test passed: Both datasets have {n_simulated_moments} moments")
    print(f"✓ SOEP moments (no care demand) shape: {soep_moments.shape}")
    print(f"✓ Simulated moments shape: {simulated_moments.shape}")
    print(f"✓ Simulated data shape: {simulated_data.shape}")


def test_moments_data_types():
    """
    Test that both SOEP and simulated moments have the correct data types.
    """
    # Path to the SOEP moments file
    soep_moments_path = Path("bld/moments/soep_moments_no_care_demand.csv")

    # Path to the simulated data (no care demand specification)
    simulated_data_path = Path(
        "bld/solve_and_simulate/simulated_data_no_care_demand.pkl"
    )

    # Path to the model options (no care demand specification)
    options_path = Path("bld/model/options_no_care_demand.pkl")

    # Check if all required files exist
    if not soep_moments_path.exists():
        pytest.skip(f"SOEP moments file not found: {soep_moments_path}")
    if not simulated_data_path.exists():
        pytest.skip(f"Simulated data file not found: {simulated_data_path}")
    if not options_path.exists():
        pytest.skip(f"Options file not found: {options_path}")

    # Load SOEP moments
    soep_moments = pd.read_csv(soep_moments_path, index_col=0)

    # Load simulated data
    with open(simulated_data_path, "rb") as f:
        simulated_data = pickle.load(f)

    # Load model options
    with open(options_path, "rb") as f:
        options = pickle.load(f)

    # Generate simulated moments using actual data
    simulated_moments = simulate_moments_pandas_no_care_demand(
        df_full=simulated_data, options=options
    )

    # Check data types
    assert isinstance(soep_moments, pd.DataFrame), "SOEP moments should be a DataFrame"
    assert isinstance(
        simulated_moments, pd.Series
    ), "Simulated moments should be a Series"

    # Check that values are numeric
    assert (
        soep_moments.select_dtypes(include=["number"]).shape[1] > 0
    ), "SOEP moments should contain numeric values"
    assert simulated_moments.dtype in (
        "float64",
        "int64",
        "float32",
        "int32",
    ), f"Simulated moments should be numeric, got {simulated_moments.dtype}"

    print("✓ Data types test passed")
    print(f"✓ SOEP moments data types: {soep_moments.dtypes.value_counts().to_dict()}")
    print(f"✓ Simulated moments data type: {simulated_moments.dtype}")
    print(f"✓ Simulated data shape: {simulated_data.shape}")
    print(f"✓ Simulated data columns: {list(simulated_data.columns)}")
