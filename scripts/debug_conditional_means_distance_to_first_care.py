"""Diagnostic script for conditional means by distance to first care demand.

Investigates:
1. Sample construction and conditioning
2. No-care-demand line kink at t=0
3. Employment drop magnitude vs other plots
"""

from pathlib import Path

import pandas as pd

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import calculate_simple_outcomes
from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.figures.publication.final.task_conditional_means_distance_to_first_care_demand import (
    _add_outcome_columns,
    _add_outcome_columns_no_care_demand,
    _compute_no_care_demand_profile,
    _identify_agents_at_least_5_years_care_demand,
    _prepare_merged_type1,
)
from caregiving.figures.publication.plotting_helpers import (
    add_distance_to_first_care_demand,
    identify_agents_by_duration,
)
from caregiving.model.shared import DEAD


def main():
    path_o = BLD / "solve_and_simulate" / "simulated_data_estimated_params_Jan7.pkl"
    path_c = BLD / "solve_and_simulate" / "simulated_data_no_care_demand.pkl"
    window = 20

    if not path_o.exists():
        print(f"Original data not found: {path_o}")
        return
    if not path_c.exists():
        print(f"No-care-demand data not found: {path_c}")
        return

    # -------------------------------------------------------------------------
    # 1. SAMPLE CONSTRUCTION
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("1. SAMPLE CONSTRUCTION")
    print("=" * 70)

    merged = _prepare_merged_type1(
        path_o, window, ever_caregivers=False, ever_care_demand=False
    )
    print("\n_prepare_merged_type1 output:")
    print(f"  Rows: {len(merged)}")
    print(f"  Unique agents: {merged['agent'].nunique()}")
    print(
        f"  Distance range: [{merged['distance_to_first_care_demand'].min()}, "
        f"{merged['distance_to_first_care_demand'].max()}]"
    )

    # Sample size by distance (for no-care-demand)
    prof_ncd = _compute_no_care_demand_profile(
        path_o, path_c, window, "work", ever_caregivers=False, ever_care_demand=False
    )
    print(f"\nNo-care-demand profile: {len(prof_ncd)} distance points")

    # Check observation count per distance for no-care-demand
    df_o = pd.read_pickle(path_o)
    df_c = pd.read_pickle(path_c)
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()
    df_o = ensure_agent_period(df_o)
    df_c = ensure_agent_period(df_c)
    type_1 = df_o[df_o["caregiving_type"] == 1]["agent"].unique()
    df_o = df_o[df_o["agent"].isin(type_1)].copy()
    df_c = df_c[df_c["agent"].isin(type_1)].copy()
    df_o_dist = add_distance_to_first_care_demand(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )
    merged_ncd = df_c.merge(dist_map, on="agent", how="inner")
    merged_ncd["distance_to_first_care_demand"] = (
        merged_ncd["period"] - merged_ncd["first_care_demand_period"]
    )
    merged_ncd = merged_ncd[
        merged_ncd["first_care_demand_period"].notna()
        & (merged_ncd["distance_to_first_care_demand"] >= -window)
        & (merged_ncd["distance_to_first_care_demand"] <= window)
    ]
    merged_ncd = _add_outcome_columns_no_care_demand(merged_ncd)

    n_per_dist = merged_ncd.groupby(
        "distance_to_first_care_demand", observed=False
    ).size()
    print("\nObservations per distance (no-care-demand) around t=0:")
    for d in range(-3, 4):
        n = n_per_dist.get(d, 0)
        print(f"  t={d:3d}: n={n}")

    # Mean age by distance
    if "age" in merged_ncd.columns:
        age_per_dist = merged_ncd.groupby(
            "distance_to_first_care_demand", observed=False
        )["age"].mean()
        print("\nMean age by distance (no-care-demand) around t=0:")
        for d in range(-3, 4):
            age = age_per_dist.get(d, float("nan"))
            print(f"  t={d:3d}: mean_age={age:.1f}")

    # -------------------------------------------------------------------------
    # 2. NO-CARE-DEMAND PROFILE AROUND t=0 (kink check)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. NO-CARE-DEMAND PROFILE AROUND t=0 (kink check)")
    print("=" * 70)

    for outcome in ["work", "working_hours_weekly", "part_time", "full_time"]:
        prof = _compute_no_care_demand_profile(
            path_o,
            path_c,
            window,
            outcome,
            ever_caregivers=False,
            ever_care_demand=False,
        )
        if len(prof) == 0:
            print(f"\n{outcome}: no data")
            continue
        around_zero = prof[
            (prof["distance_to_first_care_demand"] >= -3)
            & (prof["distance_to_first_care_demand"] <= 3)
        ].sort_values("distance_to_first_care_demand")
        print(f"\n{outcome}:")
        for _, row in around_zero.iterrows():
            print(
                f"  t={int(row['distance_to_first_care_demand']):3d}: value={row['value']:.4f}"
            )

    # -------------------------------------------------------------------------
    # 3. DURATION GROUPS AND EMPLOYMENT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DURATION GROUPS AND EMPLOYMENT RATES")
    print("=" * 70)

    merged = _add_outcome_columns(merged)
    agents_1, agents_2, agents_3, agents_4 = identify_agents_by_duration(
        merged,
        distance_col="distance_to_first_care_demand",
        duration_type="care_demand",
    )
    agents_5 = _identify_agents_at_least_5_years_care_demand(merged)

    print("\nGroup sizes:")
    print(f"  1-year (exact): {len(agents_1)}")
    print(f"  2-year (exact): {len(agents_2)}")
    print(f"  3-year (exact): {len(agents_3)}")
    print(f"  4-year (exact): {len(agents_4)}")
    print(f"  5+ year:        {len(agents_5)}")

    # Employment at t=-2, t=0, t=2 for each group and no-care-demand
    for label, agents in [
        ("no-care-demand", None),
        ("1-year", agents_1),
        ("5+ year", agents_5),
    ]:
        if agents is None:
            sub = merged_ncd
        else:
            sub = merged_ncd[merged_ncd["agent"].isin(agents)]
        if len(sub) == 0:
            print(f"\n{label}: no data")
            continue
        emp = sub.groupby("distance_to_first_care_demand", observed=False)[
            "work"
        ].mean()
        print(f"\n{label} employment rate at t=-2, 0, 2:")
        for t in [-2, 0, 2]:
            v = emp.get(t, float("nan"))
            print(f"  t={t}: {v:.4f}")

    # -------------------------------------------------------------------------
    # 4. CONDITIONING SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. CONDITIONING SUMMARY")
    print("=" * 70)
    print("""
Sample construction:
- Restrict to caregiving_type == 1 (agents who can provide informal care)
- Restrict to agents who had care_demand > 0 in ORIGINAL (implicit via dist_map)
- Distance = period - first_care_demand_period (from original)
- Window: -20 to +20 years relative to first care demand

No-care-demand line:
- Same agents (type 1, had care demand in original)
- Outcomes from NO-CARE-DEMAND simulation at same (agent, period)
- Aligned by first_care_demand_period from ORIGINAL
- No care demand in counterfactual, so no shock at t=0

Duration groups (1-5):
- Mutually exclusive subsets based on care_demand pattern in original
- 1-year: care at t=0 only, then stop
- 2-year: care at t=0,1, then stop
- ...
- 5+ year: care at t=0,1,2,3,4 (at least 5 years)
""")


if __name__ == "__main__":
    main()
