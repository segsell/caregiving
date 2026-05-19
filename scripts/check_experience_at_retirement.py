"""Check for jump/drop/discontinuity in experience when agents enter retirement.

Experience is transformed into pension points upon retirement, so we expect
a discontinuity in the raw experience (or exp_years) variable at the first
retirement period. This script loads simulated data and reports:
- Whether exp_years/experience jumps or drops at retirement entry
- Summary stats of the change (mean change, share with drop, etc.)

Run from project root with:
  PYTHONPATH=src python scripts/check_experience_at_retirement.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.model.shared import DEAD, RETIREMENT


def main() -> None:
    path_sim = BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl"
    if not path_sim.exists():
        print(f"Simulated data not found: {path_sim}")
        print("Run solve_and_simulate task first.")
        return

    df = pd.read_pickle(path_sim).reset_index()
    if "agent" not in df.columns and df.index.names:
        df = df.reset_index()
    df = df[df["health"] != DEAD].copy()
    df = df.sort_values(["agent", "period"])

    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    df["is_retired"] = df["choice"].isin(retirement_values)

    # First retirement period per agent
    first_ret = (
        df.loc[df["is_retired"]]
        .groupby("agent")["period"]
        .min()
        .rename("first_retirement_period")
    )
    first_ret = first_ret[first_ret >= 1]
    if first_ret.empty:
        print("No agents with retirement at period >= 1.")
        return

    # Experience / exp_years in period just before and at retirement
    use_exp_years = "exp_years" in df.columns
    exp_col = "exp_years" if use_exp_years else "experience"
    if exp_col not in df.columns:
        print(
            f"Neither 'exp_years' nor 'experience' found in simulated data. Columns: {list(df.columns)}"
        )
        return

    first_ret = first_ret.reset_index()
    first_ret["last_working_period"] = first_ret["first_retirement_period"] - 1

    before = df.merge(
        first_ret[["agent", "last_working_period"]],
        left_on=["agent", "period"],
        right_on=["agent", "last_working_period"],
        how="inner",
    )[["agent", exp_col]].rename(columns={exp_col: "exp_before"})

    at_ret = df.merge(
        first_ret[["agent", "first_retirement_period"]],
        left_on=["agent", "period"],
        right_on=["agent", "first_retirement_period"],
        how="inner",
    )[["agent", exp_col]].rename(columns={exp_col: "exp_at_retirement"})

    merged = before.merge(at_ret, on="agent", how="inner")
    merged["change"] = merged["exp_at_retirement"] - merged["exp_before"]
    merged["pct_change"] = (
        (merged["exp_at_retirement"] - merged["exp_before"]) / merged["exp_before"]
    ).replace([np.inf, -np.inf], np.nan)

    n = len(merged)
    print(f"Variable used: {exp_col}")
    print(f"Agents with retirement at period >= 1 and both periods observed: {n}")
    if n == 0:
        return

    print("\n--- Experience BEFORE retirement (last working period) ---")
    print(merged["exp_before"].describe())
    print("\n--- Experience AT first retirement period ---")
    print(merged["exp_at_retirement"].describe())
    print("\n--- Change (at_retirement - before) ---")
    print(merged["change"].describe())
    print(f"\nMean change: {merged['change'].mean():.4f}")
    print(f"Median change: {merged['change'].median():.4f}")
    print(f"Share with drop (change < 0): {(merged['change'] < 0).mean() * 100:.1f}%")
    print(f"Share with jump (change > 0): {(merged['change'] > 0).mean() * 100:.1f}%")
    print(f"Share with no change: {(merged['change'] == 0).mean() * 100:.1f}%")

    if use_exp_years:
        print(
            "\nConclusion: exp_years at retirement reflects transformation into pension points; "
            "use experience from LAST WORKING PERIOD (period before first retirement) for "
            "'experience at retirement entry'."
        )
    else:
        print(
            "\nConclusion: Check whether 'experience' is in years or scaled; "
            "if it jumps at retirement, use last working period for analysis."
        )


if __name__ == "__main__":
    main()
