"""Spell-length & bound-subgroup investigation (memory-conservative).

Compares the Full Beirat (capped) caregiving-leave variant to the new
No-Total-Cap variant. Produces five markdown tables (T1-T5) and a runtime
Pflegegeld cross-check. The full results and storyline narrative are
recorded in:

    docs/analysis/CAREGIVING_LEAVE_SPELL_LENGTH_INVESTIGATION.md

This is an exploratory analysis script (NOT a pytask task). Reproduce via:

    /home/sebastian/miniconda3/envs/caregiving/bin/python -u \\
        src/caregiving/sandbox/spell_length_investigation.py \\
        | tee /tmp/spell_invest.out

For each of the two simulated-data pickles the script:
  1. Loads with pd.read_pickle.
  2. IMMEDIATELY drops unused columns (keeps ~10 of however-many).
  3. Filters to alive rows (boolean indexing, no .copy).
  4. Sorts by (agent, period) once if not already sorted.
  5. Computes per-agent profile (one row per agent) via groupby('agent').agg
     and a numpy run-length-encoding step for the longest contiguous spell.
  6. del df + gc.collect() before loading the next pickle.

Memory discipline notes (peak ~4 GiB across both pickles):
- Constants (DEAD, choice sets) are hardcoded so we do NOT import
  `caregiving.model.shared` (which transitively imports `jax.numpy` ~ +500 MB).
- Each pickle is processed inside a function (`process_pickle`) so all local
  state goes out of scope on return, then a top-level `gc.collect()` runs.
- Helper masks are computed as `np.int8` numpy arrays (1 byte/row) via a tiny
  lookup table, instead of pandas Series.isin which builds bool dtype.
- After computing per-agent aggregates the source df is deleted *inside* the
  function before the function returns the small per-agent DataFrame.
- Run-length encoding uses np.cumsum(..., dtype=np.int32) (halves vs default).

A naive earlier version of this script peaked at ~17 GiB and triggered the
kernel OOM killer on a 32 GiB workstation; this version peaks at ~4 GiB.
"""

from __future__ import annotations

import gc
import resource
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root: src/caregiving/sandbox/<this file>.py -> repo root is 3 parents up.
REPO = Path(__file__).resolve().parents[3]
_PKL_FULL_BEIRAT = (
    REPO
    / "bld/solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_estimated_params.pkl"
)
_PKL_NO_TOTAL_CAP = (
    REPO
    / "bld/solve_and_simulate"
    / "simulated_data_caregiving_leave_full_beirat_no_total_cap_estimated_params.pkl"
)
PICKLES: dict[str, Path] = {
    "Full Beirat (capped)": _PKL_FULL_BEIRAT,
    "No-Total-Cap (new)": _PKL_NO_TOTAL_CAP,
}

# ---- Hardcoded constants from caregiving.model.shared (verified May 26 2026)--
DEAD = 2
JBC_PT = 1
JBC_FT = 2
LEAVE_CAP_YEARS = 3
BUCKET_SHORT_MAX = 2
BUCKET_CAP = 3
BUCKET_BEYOND_MAX = 5

FORMAL_CARE = np.array([4, 5, 6, 7], dtype=np.int8)
LIGHT_INFORMAL_CARE = np.array([8, 9, 10, 11], dtype=np.int8)
INTENSIVE_INFORMAL_CARE = np.array([12, 13, 14, 15], dtype=np.int8)
INFORMAL_CARE = np.concatenate([LIGHT_INFORMAL_CARE, INTENSIVE_INFORMAL_CARE])

RETIREMENT = np.array([0, 4, 8, 12], dtype=np.int8)
UNEMPLOYED = np.array([1, 5, 9, 13], dtype=np.int8)
PART_TIME = np.array([2, 6, 10, 14], dtype=np.int8)
FULL_TIME = np.array([3, 7, 11, 15], dtype=np.int8)
NOT_WORKING = np.concatenate([UNEMPLOYED, RETIREMENT])

# Columns we actually need; everything else is dropped right after load.
NEEDED_COLS_BASE = [
    "agent",
    "period",
    "health",
    "choice",
    "caregiving_leave_top_up",
    "care_benefits_and_costs",
    "full_leave_year_used",
    "job_before_caregiving",
    "experience",
    "assets_begin_of_period",
]
OPTIONAL_COLS = ["years_leave_used_total"]


def rss_gb() -> float:
    """Resident set size of this process in GiB (Linux: ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def log(msg: str) -> None:
    print(f"[mem={rss_gb():5.2f} GiB] {msg}", flush=True)


def isin_int8(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fast 0/1 int8 mask for `values in targets`. Uses a tiny lookup table.

    `values` is the choice column (int values 0..15). Returns int8 array same
    shape as values. Avoids constructing a temporary bool->int8 conversion
    and the overhead of pd.Series.isin.
    """
    if values.dtype != np.int8:
        values = values.astype(np.int8, copy=False)
    lut = np.zeros(64, dtype=np.int8)  # well above max choice value
    lut[targets.astype(np.int64)] = 1
    return lut[values]


def longest_informal_run_per_agent(
    agent: np.ndarray, is_informal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_agents, longest_informal_run) using only numpy.

    Assumes input arrays are aligned with df sorted by (agent, period).
    Allocates a few O(n_rows) bool arrays + O(n_runs) reduce arrays, then
    callers can free the inputs.
    """
    # Run starts where agent changes OR mask flips within an agent.
    new_agent = np.empty(agent.shape[0], dtype=bool)
    new_agent[0] = True
    np.not_equal(agent[1:], agent[:-1], out=new_agent[1:])

    flipped = np.empty(is_informal.shape[0], dtype=bool)
    flipped[0] = True
    np.not_equal(is_informal[1:], is_informal[:-1], out=flipped[1:])

    new_run = new_agent | flipped
    del flipped  # free immediately

    run_id = np.cumsum(new_run, dtype=np.int32) - 1
    del new_run

    # Length of each run (np.bincount is cheap).
    run_lengths = np.bincount(run_id).astype(np.int32, copy=False)

    # Per run: which agent and whether it is an informal-care run.
    n_runs = run_lengths.shape[0]
    run_agent = np.empty(n_runs, dtype=agent.dtype)
    run_agent[run_id] = agent
    run_is_informal = np.zeros(n_runs, dtype=bool)
    run_is_informal[run_id] = is_informal.astype(bool, copy=False)

    del run_id

    eff_lengths = np.where(run_is_informal, run_lengths, 0).astype(np.int64, copy=False)
    del run_lengths, run_is_informal

    # Reduce per agent via pandas groupby (small input now).
    s = pd.Series(eff_lengths, index=run_agent).groupby(level=0).max()
    return s.index.to_numpy(), s.to_numpy()


def process_pickle(label: str, path: Path) -> pd.DataFrame:  # noqa: PLR0915
    """Load one pickle and return the small per-agent profile DataFrame.

    Frees the source DataFrame inside this function so the caller never sees
    it. After return, gc.collect() is called.
    """
    log(f"[{label}] Loading {path.name}")
    df = pd.read_pickle(path)
    log(f"[{label}]   loaded rows={len(df):,} cols={len(df.columns)}")

    # ---- Step 1: prune columns aggressively ---------------------------------
    keep = [c for c in NEEDED_COLS_BASE if c in df.columns]
    has_ylt = "years_leave_used_total" in df.columns
    if has_ylt:
        keep.append("years_leave_used_total")
    missing = set(NEEDED_COLS_BASE) - set(keep)
    if missing:
        raise RuntimeError(f"[{label}] missing required columns: {missing}")
    df = df[keep]  # view-or-copy of the slim subset; old wide df is now dead
    gc.collect()
    log(f"[{label}]   pruned to {len(df.columns)} cols")

    # ---- Step 2: alive filter -----------------------------------------------
    alive = df["health"].to_numpy() != DEAD
    df = df.loc[alive]
    del alive
    gc.collect()
    log(f"[{label}]   alive rows={len(df):,}")

    # Drop health, no longer needed.
    df = df.drop(columns=["health"])

    # ---- Step 3: ensure (agent, period) sort once ---------------------------
    a_arr = df["agent"].to_numpy()
    p_arr = df["period"].to_numpy()
    # Cheap monotonicity check: if non-monotonic in agent OR within-agent
    # period flips, sort. We do both checks in one pass.
    same_agent = a_arr[1:] == a_arr[:-1]
    agent_monotonic = np.all(a_arr[1:] >= a_arr[:-1])
    period_monotonic_within = np.all((~same_agent) | (p_arr[1:] >= p_arr[:-1]))
    sorted_already = bool(agent_monotonic and period_monotonic_within)
    del same_agent, a_arr, p_arr
    if not sorted_already:
        log(f"[{label}]   sorting by (agent, period) ...")
        df = df.sort_values(["agent", "period"], kind="stable")
    df = df.reset_index(drop=True)
    gc.collect()
    log(f"[{label}]   sorted={sorted_already}; sort phase complete")

    # ---- Step 4: derive int8 masks (cheap, 1 byte/row) ----------------------
    choice = df["choice"].to_numpy()
    df = df.assign(
        _is_inf=isin_int8(choice, INFORMAL_CARE),
        _is_intens=isin_int8(choice, INTENSIVE_INFORMAL_CARE),
        _is_light=isin_int8(choice, LIGHT_INFORMAL_CARE),
        _is_fc=isin_int8(choice, FORMAL_CARE),
        _is_ft=isin_int8(choice, FULL_TIME),
        _is_pt=isin_int8(choice, PART_TIME),
        _is_unemp=isin_int8(choice, UNEMPLOYED),
        _is_ret=isin_int8(choice, RETIREMENT),
    )
    del choice
    df["_has_topup"] = (df["caregiving_leave_top_up"].to_numpy() > 0).astype(np.int8)
    df["_has_pfg"] = (df["care_benefits_and_costs"].to_numpy() > 0).astype(np.int8)
    df["_jbc_ft"] = (df["job_before_caregiving"].to_numpy() == JBC_FT).astype(np.int8)
    df["_jbc_pt"] = (df["job_before_caregiving"].to_numpy() == JBC_PT).astype(np.int8)
    gc.collect()
    log(f"[{label}]   int8 masks added")

    # ---- Step 5: per-agent aggregation --------------------------------------
    agg_map = {
        "_is_inf": "sum",
        "_is_intens": "sum",
        "_is_light": "sum",
        "_is_fc": "sum",
        "_is_ft": "sum",
        "_is_pt": "sum",
        "_is_unemp": "sum",
        "_is_ret": "sum",
        "_has_topup": "sum",
        "_has_pfg": "sum",
        "_jbc_ft": "sum",
        "_jbc_pt": "sum",
        "caregiving_leave_top_up": "sum",
        "care_benefits_and_costs": "sum",
        "full_leave_year_used": "max",
        "period": "count",
    }
    if has_ylt:
        agg_map["years_leave_used_total"] = "max"

    log(f"[{label}]   groupby('agent').agg ...")
    agg = df.groupby("agent", sort=True, observed=True).agg(agg_map)

    rename = {
        "_is_inf": "n_informal",
        "_is_intens": "n_intensive_informal",
        "_is_light": "n_light_informal",
        "_is_fc": "n_formal_care",
        "_is_ft": "n_ft",
        "_is_pt": "n_pt",
        "_is_unemp": "n_unemp",
        "_is_ret": "n_ret",
        "_has_topup": "n_years_with_topup",
        "_has_pfg": "n_years_with_pflegegeld",
        "_jbc_ft": "n_years_jbc_eq_2_ft",
        "_jbc_pt": "n_years_jbc_eq_1_pt",
        "caregiving_leave_top_up": "sum_topup",
        "care_benefits_and_costs": "sum_pflegegeld",
        "full_leave_year_used": "max_full_leave_year_used",
        "period": "n_alive",
    }
    if has_ylt:
        rename["years_leave_used_total"] = "max_years_leave_used_total"
    agg = agg.rename(columns=rename)

    # ---- Step 6: in-care labor split ----------------------------------------
    log(f"[{label}]   in-care labor split ...")
    inf_mask = df["_is_inf"].to_numpy().astype(bool, copy=False)
    if inf_mask.any():
        ic = df.loc[inf_mask, ["agent", "_is_ft", "_is_pt", "_is_unemp"]]
        ic_gb = ic.groupby("agent", sort=True)
        ic_split = ic_gb.agg({"_is_ft": "sum", "_is_pt": "sum", "_is_unemp": "sum"})
        ic_split.columns = ["n_ft_in_care", "n_pt_in_care", "n_unemp_in_care"]
        agg = agg.join(ic_split, how="left")
        del ic, ic_gb, ic_split
    for c in ("n_ft_in_care", "n_pt_in_care", "n_unemp_in_care"):
        if c not in agg.columns:
            agg[c] = 0
        agg[c] = agg[c].fillna(0).astype(np.int32)
    del inf_mask
    gc.collect()

    # ---- Step 7: longest contiguous informal-care spell ---------------------
    log(f"[{label}]   longest contiguous spell (RLE) ...")
    a_arr = df["agent"].to_numpy()
    is_inf_arr = df["_is_inf"].to_numpy()
    uniq_agents, longest = longest_informal_run_per_agent(a_arr, is_inf_arr)
    del a_arr, is_inf_arr
    longest_s = pd.Series(longest, index=uniq_agents, name="longest_spell")
    agg = agg.join(longest_s, how="left").fillna({"longest_spell": 0})
    agg["longest_spell"] = agg["longest_spell"].astype(np.int32)
    del uniq_agents, longest, longest_s
    gc.collect()

    # ---- Step 8: lifecycle outcomes -----------------------------------------
    log(f"[{label}]   lifecycle outcomes ...")
    last_alive_idx = df.groupby("agent")["period"].idxmax()
    last_alive = df.loc[
        last_alive_idx, ["agent", "experience", "assets_begin_of_period"]
    ].set_index("agent")
    last_alive = last_alive.rename(
        columns={
            "experience": "experience_at_last_alive",
            "assets_begin_of_period": "wealth_at_last_alive",
        }
    )
    agg = agg.join(last_alive, how="left")
    del last_alive_idx, last_alive

    is_ret_arr = df["_is_ret"].to_numpy().astype(bool, copy=False)
    if is_ret_arr.any():
        ret_df = df.loc[
            is_ret_arr, ["agent", "period", "experience", "assets_begin_of_period"]
        ]
        first_ret_idx = ret_df.groupby("agent")["period"].idxmin()
        first_ret = ret_df.loc[
            first_ret_idx, ["agent", "experience", "assets_begin_of_period", "period"]
        ].set_index("agent")
        first_ret = first_ret.rename(
            columns={
                "experience": "experience_at_retirement_entry",
                "assets_begin_of_period": "wealth_at_retirement_entry",
                "period": "period_at_retirement_entry",
            }
        )
        agg = agg.join(first_ret, how="left")
        del ret_df, first_ret_idx, first_ret
    else:
        agg["experience_at_retirement_entry"] = np.nan
        agg["wealth_at_retirement_entry"] = np.nan
        agg["period_at_retirement_entry"] = np.nan
    del is_ret_arr

    agg["variant"] = label

    # ---- Step 9: free the source DataFrame BEFORE returning -----------------
    del df
    gc.collect()
    log(f"[{label}]   profile shape={agg.shape}; source df freed")
    return agg


# =============================================================================
# Driver
# =============================================================================

print("=" * 78, flush=True)
print("Spell-length & bound-subgroup investigation (memory-conservative)", flush=True)
print("=" * 78, flush=True)
print(
    f"FT={FULL_TIME.tolist()}, PT={PART_TIME.tolist()}, "
    f"UE={NOT_WORKING.tolist()}, RT={RETIREMENT.tolist()}",  # codespell:ignore
    flush=True,
)
print(
    f"Informal={INFORMAL_CARE.tolist()}, "
    f"Intensive={INTENSIVE_INFORMAL_CARE.tolist()}, "
    f"Light={LIGHT_INFORMAL_CARE.tolist()}, "
    f"FormalCare={FORMAL_CARE.tolist()}",
    flush=True,
)
print(flush=True)

profiles: dict[str, pd.DataFrame] = {}
for idx, (label, path) in enumerate(PICKLES.items()):
    print("-" * 78, flush=True)
    print(f"[{idx+1}/{len(PICKLES)}] {label}", flush=True)
    profiles[label] = process_pickle(label, path)
    gc.collect()
    log(f"[{label}] post-collect")

# =============================================================================
# Tabulation
# =============================================================================

print(flush=True)
print("-" * 78, flush=True)
print("Building markdown tables.", flush=True)
print("-" * 78, flush=True)


def _bucket(n: int) -> str:
    if n == 0:
        return "B0 never"
    if n <= BUCKET_SHORT_MAX:
        return "B1-2 short"
    if n == BUCKET_CAP:
        return "B3 at-cap"
    if n <= BUCKET_BEYOND_MAX:
        return "B4-5 beyond-cap"
    return "B6+ very long"


for prof in profiles.values():
    prof["bucket"] = prof["n_informal"].astype(int).map(_bucket)
    prof["ever_carer"] = prof["n_informal"] > 0
    prof["ever_intensive_carer"] = prof["n_intensive_informal"] > 0
    prof["ever_leave"] = prof["n_years_with_topup"] > 0
    if "max_years_leave_used_total" in prof.columns:
        prof["hit_cap_capped"] = prof["max_years_leave_used_total"] >= LEAVE_CAP_YEARS
    else:
        prof["hit_cap_capped"] = False  # NA for new variant by construction

BUCKETS = ["B0 never", "B1-2 short", "B3 at-cap", "B4-5 beyond-cap", "B6+ very long"]
LONG_BUCKETS = ["B3 at-cap", "B4-5 beyond-cap", "B6+ very long"]
LABELS = list(profiles.keys())


def fmt_n(x) -> str:
    if pd.isna(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):,.0f}"


def fmt_pct(x) -> str:
    if pd.isna(x):
        return "--"
    return f"{100 * float(x):.2f}%"


def fmt_f(x, n_decimals: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):,.{n_decimals}f}"


# ---- T1 -------------------------------------------------------------------
print()
print("## T1. Distribution of caregivers by cumulative informal-care years")
print()
hdr = "| Bucket |"
for lbl in LABELS:
    hdr += f" {lbl}: N | {lbl}: % all | {lbl}: % ever-carer |"
print(hdr)
print("|---|" + "---:|" * (3 * len(LABELS)))
for b in BUCKETS:
    row = f"| {b} |"
    for lbl in LABELS:
        prof = profiles[lbl]
        N = int((prof["bucket"] == b).sum())
        pct_all = N / len(prof)
        n_carer = int(prof["ever_carer"].sum())
        pct_carer = (N / n_carer) if (n_carer > 0 and b != "B0 never") else float("nan")
        row += f" {fmt_n(N)} | {fmt_pct(pct_all)} | {fmt_pct(pct_carer)} |"
    print(row)
row = "| **TOTAL** |"
for lbl in LABELS:
    prof = profiles[lbl]
    N = len(prof)
    n_carer = int(prof["ever_carer"].sum())
    row += (
        f" {fmt_n(N)} | 100.00% | ever-carer N={fmt_n(n_carer)}"
        f" ({fmt_pct(n_carer / N)}) |"
    )
print(row)
row = "| **3+ years (B3+B4-5+B6+)** |"
for lbl in LABELS:
    prof = profiles[lbl]
    long_mask = prof["bucket"].isin(LONG_BUCKETS)
    N = int(long_mask.sum())
    pct_all = N / len(prof)
    n_carer = int(prof["ever_carer"].sum())
    pct_carer = N / n_carer if n_carer > 0 else float("nan")
    row += f" {fmt_n(N)} | {fmt_pct(pct_all)} | {fmt_pct(pct_carer)} |"
print(row)

# ---- T2 -------------------------------------------------------------------
print()
print("## T2. Length statistics among ever-carers")
print()
print("| Statistic | Metric | " + " | ".join(LABELS) + " |")
print("|---|---|" + "---:|" * len(LABELS))
metrics = [
    ("n_informal", "Cum informal-care years"),
    ("n_intensive_informal", "Cum intensive-inf. years"),
    ("longest_spell", "Longest contiguous spell"),
]
for col, label_metric in metrics:
    for stat in ("mean", "median", "p75", "p90", "max"):
        cells = []
        for lbl in LABELS:
            s = profiles[lbl].loc[profiles[lbl]["ever_carer"], col]
            if len(s) == 0:
                cells.append("--")
                continue
            if stat == "mean":
                v = s.mean()
            elif stat == "median":
                v = s.median()
            elif stat == "p75":
                v = s.quantile(0.75)
            elif stat == "p90":
                v = s.quantile(0.90)
            else:
                v = s.max()
            cells.append(fmt_f(v, n_decimals=2))
        print(f"| {stat} | {label_metric} | " + " | ".join(cells) + " |")

# ---- T3 -------------------------------------------------------------------
print()
print("## T3. Leave-benefit reach")
print()
print("| Metric | " + " | ".join(LABELS) + " |")
print("|---|" + "---:|" * len(LABELS))


def _hit_cap_cell(lbl: str) -> str:
    prof = profiles[lbl]
    if "max_years_leave_used_total" not in prof.columns:
        return "--<br>(column absent)"
    share = prof["hit_cap_capped"].mean()
    n_carer = int(prof["ever_carer"].sum())
    share_carer = (
        prof.loc[prof["ever_carer"], "hit_cap_capped"].mean()
        if n_carer > 0
        else float("nan")
    )
    share_leave = (
        prof.loc[prof["ever_leave"], "hit_cap_capped"].mean()
        if int(prof["ever_leave"].sum()) > 0
        else float("nan")
    )
    return (
        f"{fmt_pct(share)} of all<br>{fmt_pct(share_carer)} of ever-carer<br>"
        f"{fmt_pct(share_leave)} of EVER_LEAVE"
    )


def _ever_leave_topup(lbl: str) -> pd.Series:
    return profiles[lbl].loc[profiles[lbl]["ever_leave"], "n_years_with_topup"]


def _ever_leave_sum_topup(lbl: str) -> pd.Series:
    return profiles[lbl].loc[profiles[lbl]["ever_leave"], "sum_topup"]


rows3 = [
    (
        "Share of agents ever activating top-up",
        [fmt_pct(profiles[lbl]["ever_leave"].mean()) for lbl in LABELS],
    ),
    (
        "Mean years of top-up | EVER_LEAVE",
        [
            fmt_f(s.mean(), n_decimals=2) if len(s) else "--"
            for lbl in LABELS
            for s in (_ever_leave_topup(lbl),)
        ],
    ),
    (
        "Median years of top-up | EVER_LEAVE",
        [
            fmt_f(s.median(), n_decimals=2) if len(s) else "--"
            for lbl in LABELS
            for s in (_ever_leave_topup(lbl),)
        ],
    ),
    (
        "Max years of top-up | EVER_LEAVE",
        [
            fmt_n(int(s.max())) if len(s) else "--"
            for lbl in LABELS
            for s in (_ever_leave_topup(lbl),)
        ],
    ),
    (
        "Mean lifetime sum_topup | EVER_LEAVE (model units)",
        [
            fmt_f(s.mean(), n_decimals=3) if len(s) else "--"
            for lbl in LABELS
            for s in (_ever_leave_sum_topup(lbl),)
        ],
    ),
    (
        "HIT_CAP_capped: max years_leave_used_total >= 3",
        [_hit_cap_cell(lbl) for lbl in LABELS],
    ),
]
for label_, cells in rows3:
    print(f"| {label_} | " + " | ".join(cells) + " |")

# ---- T4 (headline) --------------------------------------------------------
print()
print("## T4. Headline: differential for 3+-informal-care subgroups")
print()
metric_specs = [
    ("n_years_with_topup", "Mean years with top-up", 2),
    ("sum_topup", "Mean lifetime sum_topup (model units)", 3),
    ("sum_pflegegeld", "Mean lifetime sum_pflegegeld (model units)", 3),
    ("ratio_ft_in_care", "Mean share FT during caregiving", 3),
    ("ratio_pt_in_care", "Mean share PT during caregiving", 3),
    ("ratio_unemp_in_care", "Mean share UE during caregiving", 3),  # codespell:ignore
    ("experience_at_retirement_entry", "Mean experience at retirement", 3),
    ("wealth_at_retirement_entry", "Mean wealth at retirement", 2),
]
for prof in profiles.values():
    denom = prof["n_informal"].replace(0, np.nan)
    prof["ratio_ft_in_care"] = prof["n_ft_in_care"] / denom
    prof["ratio_pt_in_care"] = prof["n_pt_in_care"] / denom
    prof["ratio_unemp_in_care"] = prof["n_unemp_in_care"] / denom

for bucket in LONG_BUCKETS + ["ALL 3+ years"]:
    print()
    print(f"### Bucket: {bucket}")
    print()
    ns = []
    for lbl in LABELS:
        prof = profiles[lbl]
        mask = (
            prof["bucket"].isin(LONG_BUCKETS)
            if bucket == "ALL 3+ years"
            else (prof["bucket"] == bucket)
        )
        ns.append(int(mask.sum()))
    print(
        "N = "
        + " ; ".join(f"{lbl}: {fmt_n(n)}" for lbl, n in zip(LABELS, ns, strict=True))
    )
    print()
    print("| Metric | " + " | ".join(LABELS) + " | Δ (new − capped) |")
    print("|---|" + "---:|" * (len(LABELS) + 1))
    for col, label_metric, n_decimals in metric_specs:
        cells = []
        vals = []
        for lbl in LABELS:
            prof = profiles[lbl]
            mask = (
                prof["bucket"].isin(LONG_BUCKETS)
                if bucket == "ALL 3+ years"
                else (prof["bucket"] == bucket)
            )
            s = prof.loc[mask, col]
            v = s.mean() if len(s) else float("nan")
            vals.append(v)
            cells.append(fmt_f(v, n_decimals))
        delta = vals[1] - vals[0]
        cells.append(fmt_f(delta, n_decimals))
        print(f"| {label_metric} | " + " | ".join(cells) + " |")

# ---- T5 -------------------------------------------------------------------
print()
print("## T5. Pflegegeld cross-check (should be near-identical)")
print()
print("| Metric | " + " | ".join(LABELS) + " | Δ% |")
print("|---|" + "---:|" * (len(LABELS) + 1))
cells = []
vals = []
for lbl in LABELS:
    v = float(profiles[lbl]["sum_pflegegeld"].sum())
    vals.append(v)
    cells.append(fmt_f(v, 0))
dpct = 100 * (vals[1] - vals[0]) / abs(vals[0]) if vals[0] != 0 else float("nan")
print(
    "| Total lifetime sum_pflegegeld (all agents) | "
    + " | ".join(cells)
    + f" | {fmt_f(dpct, 2)}% |"
)
cells = []
vals = []
for lbl in LABELS:
    prof = profiles[lbl]
    v = float(prof.loc[prof["ever_carer"], "sum_pflegegeld"].mean())
    vals.append(v)
    cells.append(fmt_f(v, 3))
dpct = 100 * (vals[1] - vals[0]) / abs(vals[0]) if vals[0] != 0 else float("nan")
print(
    "| Mean lifetime sum_pflegegeld | ever-carer | "
    + " | ".join(cells)
    + f" | {fmt_f(dpct, 2)}% |"
)
cells = []
vals = []
for lbl in LABELS:
    prof = profiles[lbl]
    s = prof.loc[prof["ever_carer"], "n_years_with_pflegegeld"]
    v = float(s.mean()) if len(s) else float("nan")
    vals.append(v)
    cells.append(fmt_f(v, 2))
dpct = 100 * (vals[1] - vals[0]) / abs(vals[0]) if vals[0] != 0 else float("nan")
print(
    "| Mean years with Pflegegeld | ever-carer | "
    + " | ".join(cells)
    + f" | {fmt_f(dpct, 2)}% |"
)

print()
log("Done.")
