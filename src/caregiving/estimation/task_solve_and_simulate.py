"""Solve and simulate the model for start parameters."""

import pickle
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax import tree_util
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.simulate_counterfactual import (
    simulate_counterfactual_npv,
)
from caregiving.estimation.estimation_setup import (
    load_and_setup_full_model_for_solution,
)
from caregiving.model.state_space import (
    create_state_space_functions,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.simulation.simulate import simulate_scenario
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.solve import get_solve_func_for_model

jax.config.update("jax_enable_x64", True)


# def task_solve_and_simulate_start_params(
#     path_to_solution_model: Path = BLD / "model" / "model_for_solution.pkl",
#     path_to_options: Path = BLD / "model" / "options.pkl",
#     path_to_start_params: Path = BLD / "model" / "params" / "start_params_model.yaml",
#     path_to_discrete_states: Path = BLD / "model" / "initial_conditions" / "states.pkl",
#     path_to_wealth: Path = BLD / "model" / "initial_conditions" / "wealth.csv",
#     path_to_save_solution: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "solution.pkl",
#     path_to_save_simulated_data: Annotated[Path, Product] = BLD
#     / "solve_and_simulate"
#     / "simulated_data.pkl",
#     # path_to_save_simulated_data_jax: Annotated[Path, Product] = BLD
#     # / "solve_and_simulate"
#     # / "simulated_data_jax.pkl",
# ) -> None:

#     options = pickle.load(path_to_options.open("rb"))
#     params = yaml.safe_load(path_to_start_params.open("rb"))

#     model_for_solution = load_and_setup_full_model_for_solution(
#         options, path_to_model=path_to_solution_model
#     )

#     # 1) Solve
#     solution_dict = {}
#     (
#         solution_dict["value"],
#         solution_dict["policy"],
#         solution_dict["endog_grid"],
#     ) = get_solve_func_for_model(model_for_solution)(params)
#     # value, policy, endog_grid = get_solve_func_for_model(model_for_solution)(params)

#     pickle.dump(solution_dict, path_to_save_solution.open("wb"))

#     # 2) Simulate
#     initial_states = pickle.load(path_to_discrete_states.open("rb"))
#     wealth_agents = jnp.array(pd.read_csv(path_to_wealth, usecols=["wealth"]).squeeze())

#     model_for_simulation = load_and_setup_model(
#         options=options,
#         state_space_functions=create_state_space_functions(),
#         utility_functions=create_utility_functions(),
#         utility_functions_final_period=create_final_period_utility_functions(),
#         budget_constraint=budget_constraint,
#         # shock_functions=shock_function_dict(),
#         path=path_to_solution_model,
#         sim_model=True,
#     )

#     sim_df = simulate_scenario(
#         model_for_simulation,
#         solution_endog_grid=solution_dict["endog_grid"],
#         solution_value=solution_dict["value"],
#         solution_policy=solution_dict["policy"],
#         initial_states=initial_states,
#         wealth_agents=wealth_agents,
#         params=params,
#         options=options,
#         seed=options["model_params"]["seed"],
#     )

#     # sim_df.to_csv(path_to_save_simulated_data, index=True)
#     sim_df.to_pickle(path_to_save_simulated_data)

#     # sim_df_npv = simulate_counterfactual_npv(
#     #     model_for_simulation,
#     #     solution=solution_dict,
#     #     initial_states=initial_states,
#     #     wealth_agents=wealth_agents,
#     #     params=params,
#     #     options=options,
#     #     seed=options["model_params"]["seed"],
#     # )


# ====================================================================================
# Saving/Loading utilities
# ====================================================================================

# ============================== Helpers: sizing =================================


def _is_jax_array(x: Any) -> bool:
    return isinstance(x, jax.Array)


def _nbytes(arr: Any) -> int:
    if isinstance(arr, np.ndarray):
        return arr.nbytes
    if _is_jax_array(arr):
        return int(arr.size) * arr.dtype.itemsize
    return 0


# ====================== Shrink on device BEFORE transfer ========================


def _infer_policy_indices(policy: jax.Array, *, action_axis: int = -1) -> jax.Array:
    """Return integer action indices from a policy tensor.
    If already integer, pass through; else argmax along action_axis.
    """
    if policy.dtype in (
        jnp.int8,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
    ):
        return policy  # already indices
    return jnp.argmax(policy, axis=action_axis)


def _minimal_int_dtype(max_val: int, signed: bool = False):
    if signed:
        if -128 <= max_val <= 127:
            return jnp.int8
        if -32768 <= max_val <= 32767:
            return jnp.int16
        if -2147483648 <= max_val <= 2147483647:
            return jnp.int32
        return jnp.int64
    else:
        if max_val <= 255:
            return jnp.uint8
        if max_val <= 65535:
            return jnp.uint16
        if max_val <= 4294967295:
            return jnp.uint32
        return jnp.uint64


def shrink_solution_on_device(
    solution: Dict[str, Any],
    *,
    include_value: bool = False,
    action_axis: int = -1,
) -> Dict[str, jax.Array]:
    """Produce a smaller, save-friendly dict of JAX arrays.

    - policy -> policy_idx (uint8/uint16/...)
    - endog_grid -> float32
    - value -> optional float32 (default: dropped)
    """
    out: Dict[str, jax.Array] = {}

    # policy -> indices
    pol = solution["policy"]
    pol_idx = _infer_policy_indices(pol, action_axis=action_axis)
    # compute max on device, fetch tiny scalar to host
    max_idx = int(jnp.max(pol_idx).item()) if pol_idx.size else 0
    pol_dtype = _minimal_int_dtype(max_idx, signed=False)
    out["policy_idx"] = pol_idx.astype(pol_dtype)

    # grid -> float32
    grid = solution["endog_grid"]
    out["endog_grid"] = grid.astype(jnp.float32)

    # value optional
    if include_value and "value" in solution:
        out["value"] = solution["value"].astype(jnp.float32)

    return out


# ====================== Chunked device->host saving (optional) ==================


def _save_large_array_chunked_to_npy(
    arr: jax.Array | np.ndarray,
    path: Path,
    *,
    chunk_axis: int = 0,
    target_dtype: np.dtype | None = None,
    max_bytes_per_chunk: int = 512 * 1024 * 1024,  # ~512MB per chunk
):
    """Write arr to .npy via memmap using device->host slices, avoiding huge host copies."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if _is_jax_array(arr):
        shape = tuple(arr.shape)
        dtype = np.dtype(
            target_dtype.name if target_dtype is not None else arr.dtype.name
        )
        # Prepare memmap
        mm = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        # Determine chunk size along chunk_axis
        itemsize = np.dtype(dtype).itemsize
        other_elems = (
            int(np.prod(shape) // (shape[chunk_axis] or 1)) if shape[chunk_axis] else 1
        )
        # elements per slice along axis = other_elems
        # limit chunk so that chunk_size * other_elems * itemsize <= max_bytes_per_chunk
        max_chunk_len = (
            max(1, max_bytes_per_chunk // (other_elems * itemsize))
            if other_elems > 0
            else 1
        )
        n = shape[chunk_axis]
        for start in range(0, n, max_chunk_len):
            stop = min(n, start + max_chunk_len)
            slicer = [slice(None)] * arr.ndim
            slicer[chunk_axis] = slice(start, stop)
            # fetch only the slice and cast on host
            chunk = jax.device_get(arr[tuple(slicer)])
            if target_dtype is not None and chunk.dtype != dtype:
                chunk = chunk.astype(dtype, copy=False)
            mm[tuple(slicer)] = chunk
        del mm  # flush
    else:
        # NumPy path (still memmap to be consistent)
        arr_np = (
            arr.astype(target_dtype, copy=False) if target_dtype is not None else arr
        )
        np.save(path, arr_np)


# =================== Save/Load with external .npy fallback =====================


def _tree_flatten_with_names(tree: Any) -> Tuple[Dict[str, Any], Any]:
    leaves, treedef = tree_util.tree_flatten(tree)
    named = {f"leaf_{i:04d}": leaf for i, leaf in enumerate(leaves)}
    return named, treedef


def save_solution_npz(
    solution_tree: Any,
    base_path: Path,
    *,
    big_array_threshold_bytes: int = 1_500_000_000,  # ~1.5 GB → externalize
) -> Tuple[Path, Path]:
    """Save (possibly shrunk) pytree to <base>.npz (+ meta). Very large leaves go to external .npy."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    named, treedef = _tree_flatten_with_names(solution_tree)

    npz_arrays: Dict[str, np.ndarray] = {}
    external: Dict[str, str] = {}  # name -> external .npy path
    shapes: Dict[str, Tuple[int, ...]] = {}
    dtypes: Dict[str, str] = {}

    for name, leaf in named.items():
        # Move to host leaf-by-leaf (already shrunk)
        if _is_jax_array(leaf):
            # Decide dtype downcast for host storage (float64->float32 already handled upstream)
            host_dtype = np.dtype(leaf.dtype.name)
            # If small enough, transfer whole; else externalize chunked
            est_bytes = int(leaf.size) * host_dtype.itemsize
            if est_bytes <= big_array_threshold_bytes:
                arr = jax.device_get(leaf)
                npz_arrays[name] = arr
                shapes[name] = tuple(arr.shape)
                dtypes[name] = str(arr.dtype)
            else:
                # Externalize chunked
                ext_path = base_path.parent / f"{base_path.name}__{name}.npy"
                _save_large_array_chunked_to_npy(leaf, ext_path)
                external[name] = str(ext_path.resolve())
                shapes[name] = tuple(leaf.shape)
                dtypes[name] = str(host_dtype)
        elif isinstance(leaf, np.ndarray):
            if leaf.nbytes <= big_array_threshold_bytes:
                npz_arrays[name] = leaf
                shapes[name] = tuple(leaf.shape)
                dtypes[name] = str(leaf.dtype)
            else:
                ext_path = base_path.parent / f"{base_path.name}__{name}.npy"
                _save_large_array_chunked_to_npy(leaf, ext_path)
                external[name] = str(ext_path.resolve())
                shapes[name] = tuple(leaf.shape)
                dtypes[name] = str(leaf.dtype)
        else:
            # Scalars/py objects -> pack into 0-D np array
            arr = np.asarray(leaf)
            npz_arrays[name] = arr
            shapes[name] = tuple(arr.shape)
            dtypes[name] = str(arr.dtype)

    # Save compressed npz for the non-external leaves
    npz_path = base_path.with_suffix(".npz")
    np.savez_compressed(npz_path, **npz_arrays)

    # Save metadata (includes external mapping)
    meta = {
        "format": "solution_checkpoint_v2",
        "treedef": treedef,
        "names_npz": list(npz_arrays.keys()),
        "names_external": external,  # dict name -> filepath
        "shapes": shapes,
        "dtypes": dtypes,
    }
    meta_path = base_path.with_suffix(".meta.pkl")
    with meta_path.open("wb") as f:
        pickle.dump(meta, f)

    return npz_path, meta_path


def load_solution_npz(
    base_path: Path,
    *,
    as_jax: bool = True,
    writable_numpy: bool = False,
) -> Any:
    """Load a solution saved by save_solution_npz (handles external .npy)."""
    base_path = Path(base_path)
    with (base_path.with_suffix(".meta.pkl")).open("rb") as f:
        meta = pickle.load(f)

    # Load npz leaves
    npz = np.load(base_path.with_suffix(".npz"))
    leaves_by_name: Dict[str, Any] = {}

    for name in meta["names_npz"]:
        arr = npz[name]
        if not as_jax and writable_numpy:
            arr = arr.copy()
        leaves_by_name[name] = jnp.asarray(arr) if as_jax else arr

    # Load external leaves
    for name, file_path in meta["names_external"].items():
        arr = np.load(
            file_path, mmap_mode=None
        )  # read into memory; change to 'r' if desired
        if not as_jax and writable_numpy:
            arr = arr.copy()
        leaves_by_name[name] = jnp.asarray(arr) if as_jax else arr

    # Reassemble in original leaf order: names are leaf_0000, leaf_0001, ...
    names_sorted = sorted(leaves_by_name.keys())
    leaves = [leaves_by_name[n] for n in names_sorted]
    return tree_util.tree_unflatten(meta["treedef"], leaves)
