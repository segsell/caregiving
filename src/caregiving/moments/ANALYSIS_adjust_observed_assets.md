# Analysis: `adjust_observed_assets` Function

## Overview

The `adjust_observed_assets` function (located in `dcegm/src/dcegm/asset_correction.py`) corrects observed beginning-of-period assets data for likelihood estimation. It aligns empirical survey data (which records assets without last period's income) with the model's wealth definition (which includes income from the previous period's choice).

## Function Signature and Structure

```python
def adjust_observed_assets(observed_states_dict, params, model_class):
    """
    observed_states_dict: Dictionary mapping state variable names to numpy arrays
                         Each array has shape (n_observations,)
    params: Model parameters dictionary
    model_class: Model class instance containing model functions and configuration
    """
```

## How It Works

### Step 1: Extract Wealth and Calculate End-of-Last-Period Assets

```python
wealth_int = observed_states_dict["assets_begin_of_period"]  # Shape: (n_observations,)
interest_rate = model_class.model_funcs["read_funcs"]["interest_rate"](params)
assets_end_last_period = wealth_int / (1 + interest_rate)  # Shape: (n_observations,)
```

- Takes observed beginning-of-period assets
- Discounts by interest rate to get end-of-last-period assets
- This is a vectorized operation across all observations

### Step 2: Determine Continuous States Configuration

The function checks if the model has one or two continuous states:

- **Single continuous state**: Only assets
- **Two continuous states**: Assets + another continuous state (e.g., experience)

### Step 3: Vectorized Computation via `vmap`

The function uses JAX's `vmap` to vectorize the asset adjustment computation:

#### For Single Continuous State:
```python
adjusted_assets = vmap(
    calc_beginning_of_period_assets_1cont_vec,
    in_axes=(0, 0, None, None, None, None),  # Vectorize over states and assets
)(
    observed_states_dict,      # All state variables for all observations
    assets_end_last_period,     # End-of-last-period assets for all observations
    jnp.array(0.0),            # Income shock (set to 0.0 for observed data)
    params,                     # Model parameters
    model_funcs["compute_assets_begin_of_period"],  # Budget constraint function
    False,                      # aux_outs flag
)
```

#### For Two Continuous States:
```python
adjusted_assets = vmap(
    calc_assets_beginning_of_period_2cont_vec,
    in_axes=(0, 0, 0, None, None, None, None),  # Vectorize over states, 2nd cont, assets
)(
    observed_states_dict_int,   # Discrete states (without 2nd continuous)
    second_cont_state_vars,     # Second continuous state values
    assets_end_last_period,     # End-of-last-period assets
    jnp.array(0.0),            # Income shock
    params,                     # Model parameters
    model_funcs["compute_assets_begin_of_period"],  # Budget constraint function
    False,                      # aux_outs flag
)
```

## Handling Multiple Agents/Persons

### Current Implementation

The function handles multiple agents through **vectorization**:

1. **Input Structure**: `observed_states_dict` is a dictionary where:
   - Keys are state variable names (e.g., "age", "education", "experience", "lagged_choice")
   - Values are numpy arrays of shape `(n_observations,)` where each element corresponds to one person-period observation

2. **Vectorized Processing**: 
   - All arrays in `observed_states_dict` must have the same length (n_observations)
   - `vmap` processes all observations in parallel
   - Each observation can represent a different person, a different age for the same person, or both

3. **Output**: 
   - Returns a numpy array of shape `(n_observations,)` with adjusted assets for each observation

### Example Structure

```python
observed_states_dict = {
    "age": np.array([40, 41, 42, 40, 41, ...]),           # n_observations values
    "education": np.array([0, 0, 1, 0, 1, ...]),         # n_observations values
    "experience": np.array([10, 11, 12, 5, 6, ...]),     # n_observations values
    "lagged_choice": np.array([1, 2, 3, 1, 2, ...]),     # n_observations values
    "assets_begin_of_period": np.array([100, 110, 120, ...]),  # n_observations values
    # ... other state variables
}
```

## Handling Multiple Ages

### Current Implementation

The function **does not explicitly handle age grouping**. It processes all observations together:

1. **No Age Filtering**: All observations are processed in a single vectorized call
2. **Age as a State Variable**: Age is included in `observed_states_dict` as a regular state variable
3. **Age-Dependent Computation**: The `compute_assets_begin_of_period` function receives age as part of the state vector and can use it internally

### Implications

- The function can handle observations at different ages in a single call
- Age-specific logic (if needed) must be handled within `compute_assets_begin_of_period`
- The current implementation assumes all observations are processed uniformly

## Handling Different Lagged Choices by Person and Year

### Current Problem

**The current implementation in `adjust_and_trim_wealth_data` does NOT include `lagged_choice` in `states_dict`:**

```python
# Current code (lines 2177-2183)
if adjust_wealth:
    states_dict["assets_begin_of_period"] = df["assets_begin_of_period"].values
    df["adjusted_wealth"] = adjust_observed_assets(
        observed_states_dict=states_dict,
        params=params,
        model_class=model_class,
    )
```

**Issues:**
1. `lagged_choice` is not included in `states_dict` when calling `adjust_observed_assets`
2. The `compute_assets_begin_of_period` function likely requires `lagged_choice` to correctly compute beginning-of-period assets (since income depends on previous period's choice)
3. Without `lagged_choice`, the adjustment may be incorrect or may fail

### How It Should Work

The `compute_assets_begin_of_period` function (called internally by `adjust_observed_assets`) expects:

```python
compute_assets_begin_of_period(
    **state_vec,  # Unpacks all state variables including lagged_choice
    asset_end_of_previous_period=asset_end_of_previous_period,
    income_shock_previous_period=income_shock_draw,
    params=params,
)
```

**Required State Variables:**
- All discrete state variables (age, education, experience, etc.)
- `lagged_choice`: The choice made in the previous period (critical for computing income)
- Potentially other state variables depending on the model

### Solution Requirements

To correctly adjust wealth by age and person:

1. **Include `lagged_choice` in `states_dict`**: 
   - For each person at each age, we need their `lagged_choice` from the previous period
   - This must be extracted from the data and aligned with the current period's observations

2. **Age-Person Alignment**:
   - The data should be structured such that for each person-age observation, we can look up their lagged choice
   - This may require:
     - Sorting by person ID and age
     - Creating lagged variables (e.g., `df["lagged_choice"] = df.groupby("person_id")["choice"].shift(1)`)
     - Ensuring proper alignment between current period states and lagged choice

3. **Vectorized Processing**:
   - Once `lagged_choice` is included in `states_dict`, `adjust_observed_assets` will automatically handle different lagged choices for different persons/years through vectorization
   - Each observation gets its own lagged_choice value, and the computation is vectorized across all observations

## Data Structure Requirements

### Expected DataFrame Structure

For proper wealth adjustment, the DataFrame should have:

```python
df = pd.DataFrame({
    "person_id": [1, 1, 1, 2, 2, ...],      # Person identifier
    "age": [40, 41, 42, 40, 41, ...],      # Current age
    "wealth": [100000, 110000, 120000, ...], # Observed wealth
    "choice": [1, 2, 3, 1, 2, ...],        # Current period choice
    "lagged_choice": [0, 1, 2, 0, 1, ...], # Previous period choice (needs to be created)
    "education": [0, 0, 0, 1, 1, ...],     # Education level
    "experience": [10, 11, 12, 5, 6, ...], # Experience
    # ... other state variables
})
```

### Creating Lagged Choice

The lagged choice should be created by:

```python
# Sort by person and age to ensure proper ordering
df = df.sort_values(["person_id", "age"])

# Create lagged choice within each person
df["lagged_choice"] = df.groupby("person_id")["choice"].shift(1)

# Handle first observation for each person (no lagged choice)
# Option 1: Use a default value (e.g., 0 or NOT_WORKING)
df["lagged_choice"] = df["lagged_choice"].fillna(DEFAULT_LAGGED_CHOICE)

# Option 2: Drop first observation for each person
# df = df.groupby("person_id").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
```

## Implementation Recommendations

### Sub-function Structure

Create a sub-function that:

1. **Takes the DataFrame and extracts lagged choice**:
   - Ensures data is sorted by person_id and age
   - Creates lagged_choice column if it doesn't exist
   - Handles missing lagged choices (first observation per person)

2. **Constructs states_dict with lagged_choice**:
   - Includes all required state variables
   - Includes `lagged_choice` for each observation
   - Ensures all arrays have the same length and are properly aligned

3. **Calls adjust_observed_assets**:
   - Passes the complete states_dict
   - Returns adjusted wealth

### Function Signature

```python
def adjust_wealth_by_age_and_person(
    df: pd.DataFrame,
    specs: dict,
    params: dict,
    model_class: Any,
    person_id_col: str = "person_id",
    age_col: str = "age",
    choice_col: str = "choice",
    lagged_choice_col: str = "lagged_choice",
    wealth_var: str = "wealth",
    default_lagged_choice: int = 0,
) -> pd.DataFrame:
    """
    Adjust wealth for each person at each age, accounting for lagged choices.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with person_id, age, wealth, choice, and state variables
    specs : dict
        Model specifications
    params : dict
        Model parameters
    model_class : Any
        Model class instance
    person_id_col : str
        Column name for person identifier
    age_col : str
        Column name for age
    choice_col : str
        Column name for current period choice
    lagged_choice_col : str
        Column name for lagged choice (will be created if missing)
    wealth_var : str
        Column name for wealth variable
    default_lagged_choice : int
        Default value for lagged choice when not available (first obs per person)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'adjusted_wealth' column
    """
```

## Summary

1. **Multiple Agents**: Handled via vectorization - all observations processed in parallel
2. **Multiple Ages**: Handled implicitly - age is a state variable, no explicit grouping needed
3. **Different Lagged Choices**: **Currently missing** - needs to be added to `states_dict` with proper alignment by person and age
4. **Key Fix**: Include `lagged_choice` in `states_dict` when calling `adjust_observed_assets`, ensuring proper alignment with person-age observations
