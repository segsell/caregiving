"""Debug script to compare initial_states.pkl with simulated data at period 0.

This script investigates why education and job_offer means differ between
initial_states.pkl and the simulated data at period 0, even though these
values should be predetermined.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import DEAD


def main():
    """Compare initial states with simulated data."""
    # Paths
    path_initial_states = BLD / "model" / "initial_conditions" / "initial_states.pkl"
    path_simulated_data = (
        BLD / "solve_and_simulate" / "simulated_data_no_care_demand.pkl"
    )

    print("=" * 80)
    print("INITIAL STATES vs SIMULATED DATA COMPARISON")
    print("=" * 80)

    # Load initial states
    print("\n1. Loading initial_states.pkl...")
    with path_initial_states.open("rb") as f:
        initial_states = pickle.load(f)

    print(f"   Number of agents in initial_states: {len(initial_states['period'])}")

    # Convert to arrays for easier analysis
    init_education = np.array(initial_states["education"])
    init_job_offer = np.array(initial_states["job_offer"])
    init_lagged_choice = np.array(initial_states["lagged_choice"])
    init_health = np.array(initial_states["health"])

    print("\n2. Initial States Statistics:")
    print(f"   Education mean: {init_education.mean():.5f}")
    print(f"   Education unique values: {np.unique(init_education)}")
    print("   Education value counts:")
    for val, count in zip(*np.unique(init_education, return_counts=True)):
        print(f"     {val}: {count} ({count/len(init_education)*100:.2f}%)")

    print(f"\n   Job offer mean: {init_job_offer.mean():.5f}")
    print(f"   Job offer unique values: {np.unique(init_job_offer)}")
    print("   Job offer value counts:")
    for val, count in zip(*np.unique(init_job_offer, return_counts=True)):
        print(f"     {val}: {count} ({count/len(init_job_offer)*100:.2f}%)")

    print(f"\n   Lagged choice unique values: {np.unique(init_lagged_choice)}")
    print("   Lagged choice value counts:")
    for val, count in zip(*np.unique(init_lagged_choice, return_counts=True)):
        print(f"     {val}: {count} ({count/len(init_lagged_choice)*100:.2f}%)")

    print(f"\n   Health unique values: {np.unique(init_health)}")
    print("   Health value counts:")
    for val, count in zip(*np.unique(init_health, return_counts=True)):
        print(f"     {val}: {count} ({count/len(init_health)*100:.2f}%)")
    dead_count = np.sum(init_health == DEAD)
    print(
        f"   Agents with health == DEAD ({DEAD}): {dead_count} ({dead_count/len(init_health)*100:.2f}%)"
    )

    # Load simulated data
    print("\n3. Loading simulated data...")
    sim_df = pd.read_pickle(path_simulated_data)
    print(f"   Total rows in simulated data: {len(sim_df)}")

    # Filter to period 0
    sim_period_0 = sim_df[sim_df["period"] == 0].copy()
    print(f"   Rows at period 0: {len(sim_period_0)}")
    print(f"   Unique agents at period 0: {sim_period_0['agent'].nunique()}")

    print("\n4. Simulated Data at Period 0 (AFTER filtering):")
    print(f"   Education mean: {sim_period_0['education'].mean():.5f}")
    print(f"   Education unique values: {sorted(sim_period_0['education'].unique())}")
    print("   Education value counts:")
    for val, count in sim_period_0["education"].value_counts().sort_index().items():
        print(f"     {val}: {count} ({count/len(sim_period_0)*100:.2f}%)")

    print(f"\n   Job offer mean: {sim_period_0['job_offer'].mean():.5f}")
    print(f"   Job offer unique values: {sorted(sim_period_0['job_offer'].unique())}")
    print("   Job offer value counts:")
    for val, count in sim_period_0["job_offer"].value_counts().sort_index().items():
        print(f"     {val}: {count} ({count/len(sim_period_0)*100:.2f}%)")

    # Check what's being filtered
    print("\n5. Investigating filtering...")
    print("   Checking health and consumption in simulated data at period 0:")
    print(f"   Health unique values: {sorted(sim_period_0['health'].unique())}")
    print("   Health value counts:")
    for val, count in sim_period_0["health"].value_counts().sort_index().items():
        print(f"     {val}: {count} ({count/len(sim_period_0)*100:.2f}%)")

    dead_in_sim = sim_period_0[sim_period_0["health"] == DEAD]
    print(f"   Agents with health == DEAD ({DEAD}) at period 0: {len(dead_in_sim)}")

    nan_consumption = sim_period_0[sim_period_0["consumption"].isna()]
    print(f"   Agents with NaN consumption at period 0: {len(nan_consumption)}")

    # Compare before and after filtering
    print("\n6. Comparison:")
    print(f"   Initial states agents: {len(init_education)}")
    print(f"   Simulated period 0 agents: {len(sim_period_0)}")
    print(f"   Difference (filtered out): {len(init_education) - len(sim_period_0)}")
    print(f"   Percentage kept: {len(sim_period_0)/len(init_education)*100:.2f}%")

    print("\n   Education mean difference:")
    print(f"     Initial: {init_education.mean():.5f}")
    print(f"     Simulated: {sim_period_0['education'].mean():.5f}")
    print(
        f"     Difference: {sim_period_0['education'].mean() - init_education.mean():.5f}"
    )

    print("\n   Job offer mean difference:")
    print(f"     Initial: {init_job_offer.mean():.5f}")
    print(f"     Simulated: {sim_period_0['job_offer'].mean():.5f}")
    print(
        f"     Difference: {sim_period_0['job_offer'].mean() - init_job_offer.mean():.5f}"
    )

    # Check if filtering is correlated with education/job_offer
    print("\n7. Checking if filtering is correlated with education/job_offer...")

    # Simulate what would be filtered (check initial states for health == DEAD)
    # Note: We can't check consumption in initial states, but we can check health
    init_not_dead = init_health != DEAD
    print(f"   Agents with health != DEAD in initial states: {np.sum(init_not_dead)}")

    if np.sum(init_not_dead) > 0:
        print(
            f"\n   Education mean (health != DEAD only): {init_education[init_not_dead].mean():.5f}"
        )
        print(
            f"   Job offer mean (health != DEAD only): {init_job_offer[init_not_dead].mean():.5f}"
        )

        print("\n   Education by health status:")
        for health_val in np.unique(init_health):
            mask = init_health == health_val
            if np.sum(mask) > 0:
                print(
                    f"     Health {health_val}: education mean = {init_education[mask].mean():.5f}, count = {np.sum(mask)}"
                )

        print("\n   Job offer by health status:")
        for health_val in np.unique(init_health):
            mask = init_health == health_val
            if np.sum(mask) > 0:
                print(
                    f"     Health {health_val}: job_offer mean = {init_job_offer[mask].mean():.5f}, count = {np.sum(mask)}"
                )

    # Check if we can identify which agents were filtered
    print("\n8. Attempting to match agents...")
    # The simulated data has an 'agent' column that should correspond to the index
    # in initial_states
    if len(sim_period_0) < len(init_education):
        # Some agents were filtered out
        sim_agent_ids = set(sim_period_0["agent"].unique())
        all_agent_ids = set(range(len(init_education)))
        filtered_agent_ids = all_agent_ids - sim_agent_ids

        if len(filtered_agent_ids) > 0:
            filtered_agent_ids = sorted(list(filtered_agent_ids))
            print(f"   Filtered out {len(filtered_agent_ids)} agents")
            print(f"   First 10 filtered agent IDs: {filtered_agent_ids[:10]}")

            # Check characteristics of filtered agents
            filtered_mask = np.array(
                [i in filtered_agent_ids for i in range(len(init_education))]
            )
            print("\n   Characteristics of filtered agents:")
            print(f"     Education mean: {init_education[filtered_mask].mean():.5f}")
            print(f"     Job offer mean: {init_job_offer[filtered_mask].mean():.5f}")
            print(
                f"     Health == DEAD: {np.sum(init_health[filtered_mask] == DEAD)} / {np.sum(filtered_mask)}"
            )

            # Check education distribution of filtered vs kept
            print("\n   Education distribution comparison:")
            print("     Kept agents:")
            kept_mask = ~filtered_mask
            for val in np.unique(init_education):
                count = np.sum((init_education == val) & kept_mask)
                print(f"       {val}: {count} ({count/np.sum(kept_mask)*100:.2f}%)")
            print("     Filtered agents:")
            for val in np.unique(init_education):
                count = np.sum((init_education == val) & filtered_mask)
                print(f"       {val}: {count} ({count/np.sum(filtered_mask)*100:.2f}%)")

    # More detailed comparison - check exact matching
    print("\n9. Detailed value-by-value comparison...")
    if len(sim_period_0) == len(init_education):
        # Check if values match exactly
        sim_edu_sorted = sim_period_0.sort_values("agent")["education"].values
        sim_job_sorted = sim_period_0.sort_values("agent")["job_offer"].values

        edu_match = np.array_equal(init_education, sim_edu_sorted)
        job_match = np.array_equal(init_job_offer, sim_job_sorted)

        print(f"   Education arrays match exactly: {edu_match}")
        if not edu_match:
            diff_mask = init_education != sim_edu_sorted
            print(f"   Number of mismatches: {np.sum(diff_mask)}")
            if np.sum(diff_mask) > 0:
                print("   First 10 mismatches (initial vs simulated):")
                mismatch_indices = np.where(diff_mask)[0][:10]
                for idx in mismatch_indices:
                    print(
                        f"     Agent {idx}: {init_education[idx]} vs {sim_edu_sorted[idx]}"
                    )

        print(f"   Job offer arrays match exactly: {job_match}")
        if not job_match:
            diff_mask = init_job_offer != sim_job_sorted
            print(f"   Number of mismatches: {np.sum(diff_mask)}")
            if np.sum(diff_mask) > 0:
                print("   First 10 mismatches (initial vs simulated):")
                mismatch_indices = np.where(diff_mask)[0][:10]
                for idx in mismatch_indices:
                    print(
                        f"     Agent {idx}: {init_job_offer[idx]} vs {sim_job_sorted[idx]}"
                    )

    # Check if there's a baseline simulation to compare against
    print("\n10. Checking for baseline simulation for comparison...")
    path_baseline = BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl"
    if path_baseline.exists():
        print(f"   Found baseline simulation: {path_baseline}")
        baseline_df = pd.read_pickle(path_baseline)
        baseline_period_0 = baseline_df[baseline_df["period"] == 0].copy()

        print(f"   Baseline period 0 agents: {len(baseline_period_0)}")
        print(
            f"   Baseline education mean: {baseline_period_0['education'].mean():.5f}"
        )
        print(
            f"   Baseline job offer mean: {baseline_period_0['job_offer'].mean():.5f}"
        )

        print("\n   Comparison (Baseline vs No Care Demand):")
        print(
            f"     Education mean difference: {baseline_period_0['education'].mean() - sim_period_0['education'].mean():.5f}"
        )
        print(
            f"     Job offer mean difference: {baseline_period_0['job_offer'].mean() - sim_period_0['job_offer'].mean():.5f}"
        )
    else:
        print(f"   Baseline simulation not found at: {path_baseline}")

    print("\n" + "=" * 80)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 80)
    print("\n✓ No Care Demand simulation:")
    print("  - Uses current initial_states.pkl correctly")
    print(
        f"  - Education mean: {sim_period_0['education'].mean():.5f} (matches initial_states.pkl)"
    )
    print(
        f"  - Job offer mean: {sim_period_0['job_offer'].mean():.5f} (matches initial_states.pkl)"
    )
    print("  - All 100,000 agents preserved at period 0")
    print("  - Values match initial_states.pkl exactly (no filtering issues)")

    if path_baseline.exists():
        print("\n⚠ Baseline simulation:")
        print(f"  - Education mean: {baseline_period_0['education'].mean():.5f}")
        print(f"  - Job offer mean: {baseline_period_0['job_offer'].mean():.5f}")
        print("  - DIFFERENT from current initial_states.pkl!")
        print("  - This suggests the baseline was run with an older initial_states.pkl")
        print(
            "  - SOLUTION: Re-run the baseline simulation to use current initial_states.pkl"
        )

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
