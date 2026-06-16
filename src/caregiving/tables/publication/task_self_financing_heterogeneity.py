"""Self-financing heterogeneity tables (4 configurations).

Pytask tasks that build the heterogeneity self-financing tables agreed in the
May 18-27 and June 12 design conversations. Four formula configurations
(MVPF-honest [headline], Pension-excluded, Existing-published-comparator,
Legacy-hybrid back-comparability anchor; the former "Fabian-style" config (ii)
was dropped as incoherent), each in two discounting variants (discounted at
``R_GOV_ANNUAL = 1%`` and undiscounted), evaluated on the heterogeneity grid
(6 reforms x 2 education x 4 first-care-demand age bins) under three
Fabian-symmetric perspective rows (care demand periods, active caregiving
periods, all periods).

Structure
---------

Pytask tasks:

1. ``task_aggregate_self_financing_heterogeneity`` -- loads the seven policy
   pickles via :mod:`_self_financing_ledger`, runs the partner-flow
   invariance assertion, and writes the long-format aggregate pickle
   ``bld/derived/self_financing_heterogeneity_aggregates.pkl``.
2. ``task_write_self_financing_heterogeneity_tables`` (and the
   ``_intersection_mask`` twin) -- read the pickle and emit one LaTeX file
   per config (4 configs x 2 masks = 8 files, each with discounted +
   undiscounted panels and a notes block) under
   ``bld/tables/publication/self_financing_heterogeneity/``.
3. ``task_write_self_financing_compact_tables`` -- the two compact lead
   tables (degrees overview; Config (i) decomposition with F mechanical/
   behavioral memo rows, the excluded health-SSC memo, and absolute EUR
   surplus/outlay rows).
4. ``task_compile_self_financing_report`` -- assembles ``sf_report.tex``
   and compiles everything into ``sf_report.pdf`` via latexmk.

The aggregator stores, per (policy, perspective, mask, discounting,
subgroup) cell, both masked EUR totals (``X_total``; used by the
totals-normalized configs i/iii/iv) and per-ever-caregiver averages
(``X_avg``; used only by the legacy-convention Config (v)).

Config (v) Legacy hybrid: a strict reproduction of the existing aggregate
``task_self_financing.py`` (which is read-only on the master fiscal table).
A, B (S1 broad), C, G are pulled from active-caregiving rows; D from
all-rows; per-policy-caregiver averages. The three perspective-row labels in
any (v) sub-panel print identical numbers because the hybrid mask is
fixed-internal (ignores the rendered perspective). Used as the hard-fail
back-comparability anchor on ``Config (v) discounted, Reform / total,
mask_v1`` against the published 152.4 percent.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import INFORMAL_CARE
from caregiving.tables.publication._self_financing_ledger import (
    AGE_BIN_EDGES,
    NEVER_LABEL,
    PATHS_BY_KEY,
    POLICY_KEYS,
    POLICY_LABELS_SHORT,
    build_ledger,
    build_subgroup_table,
    load_slim,
    load_specs,
)

PERSPECTIVES: tuple[str, ...] = ("care_demand", "active_caregiving", "all_periods")

# Government discount rate sweep. Each label maps to an annual real rate; the
# per-period weight is (1/(1+rate))**period (rate 0 == undiscounted face value).
# Computed in a single aggregation pass. ``disc2p25`` (the model's own
# ``interest_rate``) is the internally-consistent primary headline; ``disc1`` is
# the legacy Stuermer-Heiber 1% convention; ``disc3`` is a conventional social
# discount rate.
DISCOUNT_RATES: dict[str, float] = {
    "undiscounted": 0.0,
    "disc1": 0.01,
    "disc2p25": 0.0225,
    "disc3": 0.03,
}
DISCOUNTINGS: tuple[str, ...] = tuple(DISCOUNT_RATES)
PRIMARY_DISCOUNTING: str = "disc2p25"
LEGACY_DISCOUNTING: str = "disc1"
DISCOUNTING_LABELS: dict[str, str] = {
    "undiscounted": "undiscounted",
    "disc1": "discounted (1\\%)",
    "disc2p25": "discounted (2.25\\%, model $r$)",
    "disc3": "discounted (3\\%)",
}
MASK_CHOICES: tuple[str, ...] = ("v1", "intersection")

CONFIG_ORDER: tuple[str, ...] = ("i", "iii", "iv", "v")
CONFIG_LABELS: dict[str, str] = {
    "i": "Config (i) MVPF-honest (S4 LTC-symmetric) -- HEADLINE",
    "iii": "Config (iii) Pension-excluded (S4 LTC-symmetric)",
    "iv": "Config (iv) Existing-published comparator (S1 broad)",
    "v": "Config (v) Legacy hybrid back-comparability anchor (S1 broad)",
}

PERSPECTIVE_LABELS: dict[str, str] = {
    "care_demand": "Care demand periods",
    "active_caregiving": "Active caregiving periods",
    "all_periods": "All periods",
}

COUNTERFACTUALS: tuple[str, ...] = (
    "full_beirat",
    "full_beirat_no_total_cap",
    "partial_beirat",
    "normal_leave",
    "norwegian_pg",
    "norwegian_no_pg",
)

COMPONENT_COLS: tuple[str, ...] = (
    "A_at_eur",
    "B_income_tax_at_eur",
    "B_pension_ssc_at_eur",
    "B_ue_ssc_at_eur",
    "B_ltc_ssc_at_eur",
    "B_health_ssc_at_eur",
    "C_at_eur",
    "D_at_eur",
    "E_at_eur",
    "F_at_eur",
    "Ftilde_at_eur",
    "G_at_eur",
    "partner_ssc_at_eur",
    "partner_pension_at_eur",
    "T_total_at_eur",
)
COMPONENT_BASENAME: dict[str, str] = {
    "A_at_eur": "A",
    "B_income_tax_at_eur": "B_income_tax",
    "B_pension_ssc_at_eur": "B_pension_ssc",
    "B_ue_ssc_at_eur": "B_ue_ssc",
    "B_ltc_ssc_at_eur": "B_ltc_ssc",
    "B_health_ssc_at_eur": "B_health_ssc",
    "C_at_eur": "C",
    "D_at_eur": "D",
    "E_at_eur": "E",
    "F_at_eur": "F",
    "Ftilde_at_eur": "Ftilde",
    "G_at_eur": "G",
    "partner_ssc_at_eur": "partner_ssc",
    "partner_pension_at_eur": "partner_pension",
    "T_total_at_eur": "T_total",
}
# Each component is stored twice per aggregator row: as a per-ever-caregiver
# average (``X_avg``; used by the legacy-convention Config (v)) and as the
# masked EUR total (``X_total``; used by the totals-based Configs i/iii/iv).
COMPONENT_AVG_RENAME: dict[str, str] = {
    col: f"{base}_avg" for col, base in COMPONENT_BASENAME.items()
}
COMPONENT_TOTAL_RENAME: dict[str, str] = {
    col: f"{base}_total" for col, base in COMPONENT_BASENAME.items()
}

# Tolerance for the partner-flow exogeneity assertion (relative deviation of
# undiscounted all-rows totals of partner SSC / partner pension vs baseline).
PARTNER_FLOW_TOLERANCE: float = 0.005

EXISTING_REFORM_SF_PCT: float = 152.4
SANITY_TOLERANCE_PCT: float = 0.5


# =============================================================================
# Aggregator
# =============================================================================


def _attach_baseline_care_demand_single(
    ledger: pd.DataFrame, base_cd: pd.DataFrame
) -> pd.DataFrame:
    """Attach ``mask_baseline_care_demand`` to a single ledger.

    ``base_cd`` is the small baseline frame
    ``[agent, period, mask_baseline_care_demand]`` (extracted once from the
    baseline ledger). False where no matching baseline row exists (the
    agent died before that period in baseline). Used by the
    intersection-mask variant.
    """
    merged = ledger.merge(base_cd, on=["agent", "period"], how="left")
    for col in ("mask_baseline_care_demand", "mask_baseline_care_demand_lagged"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
    return merged


def _attach_active_mask_single(
    ledger: pd.DataFrame, cf_mask: pd.DataFrame, cf: str
) -> pd.DataFrame:
    """Broadcast one counterfactual's active-caregiving mask onto a ledger.

    ``cf_mask`` is the small frame ``[agent, period, mask_active_<cf>]``.
    Cells with no matching source row (mortality timing) become False.
    """
    merged = ledger.merge(cf_mask, on=["agent", "period"], how="left")
    merged[f"mask_active_{cf}"] = merged[f"mask_active_{cf}"].fillna(False).astype(bool)
    return merged


def _attach_care_demand_mask_single(
    ledger: pd.DataFrame, cf_mask: pd.DataFrame, cf: str
) -> pd.DataFrame:
    """Broadcast one counterfactual's care-demand mask onto a ledger.

    ``cf_mask`` is the small frame ``[agent, period, mask_<cf>_care_demand]``.
    Used so the BASELINE column can be clipped by the counterfactual's care
    demand for the Chapter-3 symmetric care-demand intersection (Option B).
    Cells with no matching source row become False.
    """
    merged = ledger.merge(cf_mask, on=["agent", "period"], how="left")
    merged[f"mask_{cf}_care_demand"] = (
        merged[f"mask_{cf}_care_demand"].fillna(False).astype(bool)
    )
    return merged


def _row_mask(
    ledger: pd.DataFrame,
    perspective: str,
    mask_choice: str,
    mask_source: str,
) -> np.ndarray:
    """Return the boolean per-row mask for the given combo."""
    if perspective == "care_demand":
        # "self" = the ledger's own care-demand state (Chapter 2 per-arm, and
        # the policy column under intersection); a policy key = that
        # counterfactual's care-demand state broadcast onto this ledger (used so
        # the BASELINE column can be clipped by the counterfactual's care demand
        # -> Chapter 3 symmetric intersection, Option B).
        col = (
            "mask_care_demand"
            if mask_source == "self"
            else f"mask_{mask_source}_care_demand"
        )
        m = ledger[col].values.astype(bool)
    elif perspective == "all_periods":
        m = ledger["mask_all"].values.astype(bool)
    elif perspective == "active_caregiving":
        # "self" = the ledger's own lagged-choice caregiving mask (legacy
        # convention used by Config (v)); a policy key = that
        # counterfactual's mask broadcast onto this ledger.
        col = (
            "mask_active_caregiving"
            if mask_source == "self"
            else f"mask_active_{mask_source}"
        )
        m = ledger[col].values.astype(bool)
    else:  # pragma: no cover - defended by enums above
        raise ValueError(f"Unknown perspective: {perspective}")

    if mask_choice == "intersection":
        # All-periods is the lifecycle measure; survival is policy-invariant
        # (verified), so the survival-aligned common support IS the full
        # lifecycle -> do NOT clip it by baseline care demand.
        if perspective == "all_periods":
            return m
        # Care-demand and active-caregiving rows are clipped by the baseline
        # care-demand state. The active-caregiving mask is defined on the LAGGED
        # choice (the agent gave informal care last period), so it must be
        # intersected with the LAGGED baseline care demand; contemporaneous would
        # spuriously drop spell-boundary cells (verified: contemporaneous drops
        # ~46% of active cells, ~42 pp pure timing; lagged drops only the ~3.7%
        # genuinely care-demand-non-invariant cells). The care-demand row is a
        # contemporaneous quantity and keeps the contemporaneous baseline mask.
        base_col = (
            "mask_baseline_care_demand_lagged"
            if perspective == "active_caregiving"
            else "mask_baseline_care_demand"
        )
        m = m & ledger[base_col].values.astype(bool)
    return m


def _ever_caregiver_per_agent(ledger: pd.DataFrame) -> pd.Series:
    """Per-agent boolean: ever has ``choice in INFORMAL_CARE`` in this policy.

    Matches the existing pipeline's ``n_caregivers`` definition (see
    :func:`caregiving.tables.publication.task_fiscal._avg_outcomes_per_caregiver`).
    """
    informal_codes = np.asarray(INFORMAL_CARE).ravel()
    is_informal_choice = ledger["choice"].isin(informal_codes)
    return is_informal_choice.groupby(ledger["agent"]).max()


_BIN_LABELS_ORDERED: tuple[str, ...] = tuple(b for _, _, b in AGE_BIN_EDGES)


def _subgroup_keys() -> list[tuple[str, str]]:
    """All (education, age_bin) combos written into the aggregator pickle.

    Includes the totals ("all", "all"), per-edu totals ("low/high", "all"),
    and the heterogeneity cells (edu, age_bin in 40-49 / 50-59 / 60-69).
    The heterogeneity cells exclude never-demand agents by definition; the
    "all"-bin rows include them.
    """
    keys: list[tuple[str, str]] = [("all", "all")]
    for edu in ("low", "high"):
        keys.append((edu, "all"))
        for ab in _BIN_LABELS_ORDERED:
            keys.append((edu, ab))
    return keys


def _subgroup_agent_filter(
    sg: pd.DataFrame, education: str, age_bin: str
) -> np.ndarray:
    """Boolean per-agent mask selecting the (edu, age_bin) subgroup."""
    sel = np.ones(len(sg), dtype=bool)
    if education != "all":
        sel = sel & (sg["education_label"].values == education)
    if age_bin != "all":
        sel = sel & (sg["age_bin"].values == age_bin)
    return sel


def _build_subgroup_row_masks(
    ledger: pd.DataFrame, sg: pd.DataFrame
) -> dict[tuple[str, str], tuple[np.ndarray, int, np.ndarray]]:
    """Pre-compute per-subgroup boolean row masks and per-subgroup agent
    counts. Returned dict maps ``(education, age_bin)`` to
    ``(row_in_sg_bool, n_total_in_subgroup, sg_agent_index)``.
    """
    agent_ids = ledger["agent"].values
    out: dict[tuple[str, str], tuple[np.ndarray, int, np.ndarray]] = {}
    for education, age_bin in _subgroup_keys():
        agent_in_sg = _subgroup_agent_filter(sg, education, age_bin)
        sg_agent_index = np.asarray(sg.index[agent_in_sg])
        if sg_agent_index.size == 0:
            continue
        row_in_sg = np.isin(agent_ids, sg_agent_index)
        out[(education, age_bin)] = (
            row_in_sg,
            int(sg_agent_index.size),
            sg_agent_index,
        )
    return out


def _aggregate_for_policy(
    ledger: pd.DataFrame,
    sg: pd.DataFrame,
    *,
    policy: str,
    mask_sources_by_perspective: dict[str, tuple[str, ...]],
) -> list[dict]:
    """Aggregate all (perspective, mask_source, mask_choice, discounting)
    combos for one policy across all subgroups. Pre-computes per-subgroup
    masks and per-agent ever-caregiver flags ONCE.
    """
    subgroup_masks = _build_subgroup_row_masks(ledger, sg)
    ever_provider = _ever_caregiver_per_agent(ledger)
    period = ledger["period"].values.astype(float)
    # Per-rate discount weights computed once: (1/(1+rate))**period; rate 0 == 1.
    discount_weights = {
        label: np.power(1.0 / (1.0 + rate), period)
        for label, rate in DISCOUNT_RATES.items()
    }
    component_arrays = {col: ledger[col].values for col in COMPONENT_COLS}

    records: list[dict] = []
    for perspective in PERSPECTIVES:
        for mask_source in mask_sources_by_perspective[perspective]:
            for mask_choice in MASK_CHOICES:
                row_mask = _row_mask(ledger, perspective, mask_choice, mask_source)
                row_mask_f = row_mask.astype(float)
                for discounting in DISCOUNTINGS:
                    weight = discount_weights[discounting]
                    for (education, age_bin), (
                        row_in_sg,
                        n_total_in_subgroup,
                        sg_agent_index,
                    ) in subgroup_masks.items():
                        weights_here = weight * row_mask_f * row_in_sg.astype(float)
                        provider_in_sg = ever_provider.reindex(sg_agent_index).fillna(
                            False
                        )
                        n_ever_caregivers = int(provider_in_sg.sum())
                        record: dict = {
                            "policy": policy,
                            "perspective": perspective,
                            "mask_source": mask_source,
                            "mask_choice": mask_choice,
                            "discounting": discounting,
                            "education": education,
                            "age_bin": age_bin,
                            "n_total_in_subgroup": n_total_in_subgroup,
                            "n_ever_caregivers": n_ever_caregivers,
                            "n_masked_cells": int(np.sum(row_mask & row_in_sg)),
                        }
                        denom = n_ever_caregivers if n_ever_caregivers > 0 else np.nan
                        for col in COMPONENT_COLS:
                            total = float(np.dot(component_arrays[col], weights_here))
                            record[COMPONENT_TOTAL_RENAME[col]] = total
                            record[COMPONENT_AVG_RENAME[col]] = (
                                total / denom if denom and denom > 0 else np.nan
                            )
                        records.append(record)
    return records


def _assert_partner_flows_from_sums(
    partner_sums: dict[str, tuple[float, float]],
) -> None:
    """Hard assertion: partner-side fiscal flows are policy-invariant.

    Partner behaviour is exogenous (same RNG paths across policies), so the
    undiscounted all-rows totals of partner SSC and partner pension must
    agree with baseline up to ``PARTNER_FLOW_TOLERANCE`` for every policy.
    Operates on the per-policy ``(partner_ssc_total, partner_pension_total)``
    sums accumulated during the incremental aggregation (so no ledger needs
    to be held in memory for this check). Prints the comparison table and
    raises on violation.
    """
    base_ssc, base_pen = partner_sums["baseline"]
    print(
        "[SF heterogeneity] Partner-flow invariance check "
        f"(tolerance {PARTNER_FLOW_TOLERANCE:.1%}):",
        flush=True,
    )
    print(
        f"  baseline: partner_ssc = {base_ssc:,.0f} EUR, "
        f"partner_pension = {base_pen:,.0f} EUR",
        flush=True,
    )
    violations: list[str] = []
    for key, (ssc, pen) in partner_sums.items():
        if key == "baseline":
            continue
        dev_ssc = abs(ssc - base_ssc) / base_ssc if base_ssc else 0.0
        dev_pen = abs(pen - base_pen) / base_pen if base_pen else 0.0
        status = (
            "OK"
            if dev_ssc <= PARTNER_FLOW_TOLERANCE and dev_pen <= PARTNER_FLOW_TOLERANCE
            else "FAIL"
        )
        print(
            f"  {key}: partner_ssc dev = {dev_ssc:.4%}, "
            f"partner_pension dev = {dev_pen:.4%} -- {status}",
            flush=True,
        )
        if status == "FAIL":
            violations.append(key)
    if violations:
        raise RuntimeError(
            "Partner-side flows are NOT policy-invariant for "
            f"{violations}. Partner behaviour is exogenous by design; "
            "deltas of partner SSC / partner pension should be ~0. "
            "Investigate the simulation output before building SF tables."
        )


def _ms_by_perspective(policy: str) -> dict[str, tuple[str, ...]]:
    """Mask-source tuples per perspective for one policy.

    Baseline carries ``self`` plus every counterfactual under the
    active-caregiving perspective: ``self`` feeds the Config (v) legacy
    convention (each policy masked by its OWN informal-care rows), while
    each counterfactual mask source feeds the symmetric active-caregiving
    rows of Configs (i)/(iii)/(iv). Counterfactuals carry only their own
    mask source.
    """
    if policy == "baseline":
        return {
            # ``self`` feeds Chapter 2 (per-arm care demand) and Config (v); the
            # counterfactual sources feed the Chapter-3 symmetric care-demand
            # intersection (Option B), where the baseline is clipped by each
            # counterfactual's care demand.
            "care_demand": ("self", *COUNTERFACTUALS),
            "active_caregiving": ("self", *COUNTERFACTUALS),
            "all_periods": ("self",),
        }
    return {
        # ``self`` = Chapter 2 (own care demand); ``policy`` = Chapter 3
        # intersection (own care demand, looked up under the policy's own key so
        # the symmetric pair shares a mask source).
        "care_demand": ("self", policy),
        "active_caregiving": (policy,),
        "all_periods": ("self",),
    }


def _build_aggregate_long_incremental(
    inputs, sg: pd.DataFrame, baseline_ledger: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate every policy with at most one counterfactual ledger
    resident at a time (the baseline ledger stays resident throughout).

    This is the OOM-safe replacement for holding all seven ledgers plus a
    full copy. The counterfactual ledgers are loaded, aggregated, reduced
    to a small active-mask frame for the baseline's symmetric rows, and
    freed before the next pickle is touched. Numerics are identical to the
    all-in-memory path: the same masks, the same mask sources, the same
    per-policy aggregation.
    """
    import gc

    base_cd = baseline_ledger[
        ["agent", "period", "mask_care_demand", "mask_care_demand_lagged"]
    ].rename(
        columns={
            "mask_care_demand": "mask_baseline_care_demand",
            "mask_care_demand_lagged": "mask_baseline_care_demand_lagged",
        }
    )

    rows: list[dict] = []
    partner_sums: dict[str, tuple[float, float]] = {}
    cf_active_masks: dict[str, pd.DataFrame] = {}
    cf_care_demand_masks: dict[str, pd.DataFrame] = {}

    for cf in COUNTERFACTUALS:
        slim = load_slim(PATHS_BY_KEY[cf])
        ledger = build_ledger(slim, inputs, cf, sg)
        del slim
        gc.collect()

        ledger = _attach_baseline_care_demand_single(ledger, base_cd)
        # The counterfactual's own active + care-demand masks, addressed under
        # the counterfactual's own key (so the symmetric-intersection pair shares
        # a mask source with the broadcast baseline row).
        ledger[f"mask_active_{cf}"] = ledger["mask_active_caregiving"]
        ledger[f"mask_{cf}_care_demand"] = ledger["mask_care_demand"]

        rows.extend(
            _aggregate_for_policy(
                ledger,
                sg,
                policy=cf,
                mask_sources_by_perspective=_ms_by_perspective(cf),
            )
        )
        partner_sums[cf] = (
            float(ledger["partner_ssc_at_eur"].sum()),
            float(ledger["partner_pension_at_eur"].sum()),
        )
        cf_active_masks[cf] = ledger[
            ["agent", "period", "mask_active_caregiving"]
        ].rename(columns={"mask_active_caregiving": f"mask_active_{cf}"})
        cf_care_demand_masks[cf] = ledger[
            ["agent", "period", "mask_care_demand"]
        ].rename(columns={"mask_care_demand": f"mask_{cf}_care_demand"})
        del ledger
        gc.collect()

    # Baseline last: it needs its own care-demand column and every
    # counterfactual's active + care-demand masks broadcast onto it.
    baseline_ledger = _attach_baseline_care_demand_single(baseline_ledger, base_cd)
    for cf in COUNTERFACTUALS:
        baseline_ledger = _attach_active_mask_single(
            baseline_ledger, cf_active_masks[cf], cf
        )
        baseline_ledger = _attach_care_demand_mask_single(
            baseline_ledger, cf_care_demand_masks[cf], cf
        )
        del cf_active_masks[cf]
        del cf_care_demand_masks[cf]
    gc.collect()

    rows.extend(
        _aggregate_for_policy(
            baseline_ledger,
            sg,
            policy="baseline",
            mask_sources_by_perspective=_ms_by_perspective("baseline"),
        )
    )
    partner_sums["baseline"] = (
        float(baseline_ledger["partner_ssc_at_eur"].sum()),
        float(baseline_ledger["partner_pension_at_eur"].sum()),
    )
    del baseline_ledger
    gc.collect()

    _assert_partner_flows_from_sums(partner_sums)
    return pd.DataFrame(rows)


def _build_aggregate_long(
    ledgers: dict[str, pd.DataFrame], sg: pd.DataFrame
) -> pd.DataFrame:
    """Run all (policy x perspective x mask_source x mask_choice x discounting)
    combinations and return the long-format aggregate DataFrame.

    All-in-memory variant retained for tests on small fixtures. The
    production task uses :func:`_build_aggregate_long_incremental`, which is
    OOM-safe on the full ~3.9 GB pickles. Both paths produce identical
    output.
    """
    base_cd = ledgers["baseline"][
        ["agent", "period", "mask_care_demand", "mask_care_demand_lagged"]
    ].rename(
        columns={
            "mask_care_demand": "mask_baseline_care_demand",
            "mask_care_demand_lagged": "mask_baseline_care_demand_lagged",
        }
    )
    cf_active_masks = {
        cf: ledgers[cf][["agent", "period", "mask_active_caregiving"]].rename(
            columns={"mask_active_caregiving": f"mask_active_{cf}"}
        )
        for cf in COUNTERFACTUALS
    }
    cf_care_demand_masks = {
        cf: ledgers[cf][["agent", "period", "mask_care_demand"]].rename(
            columns={"mask_care_demand": f"mask_{cf}_care_demand"}
        )
        for cf in COUNTERFACTUALS
    }

    rows: list[dict] = []
    partner_sums: dict[str, tuple[float, float]] = {}
    for policy in POLICY_KEYS:
        ledger = _attach_baseline_care_demand_single(ledgers[policy], base_cd)
        if policy == "baseline":
            for cf in COUNTERFACTUALS:
                ledger = _attach_active_mask_single(ledger, cf_active_masks[cf], cf)
                ledger = _attach_care_demand_mask_single(
                    ledger, cf_care_demand_masks[cf], cf
                )
        else:
            ledger[f"mask_active_{policy}"] = ledger["mask_active_caregiving"]
            ledger[f"mask_{policy}_care_demand"] = ledger["mask_care_demand"]
        rows.extend(
            _aggregate_for_policy(
                ledger,
                sg,
                policy=policy,
                mask_sources_by_perspective=_ms_by_perspective(policy),
            )
        )
        partner_sums[policy] = (
            float(ledger["partner_ssc_at_eur"].sum()),
            float(ledger["partner_pension_at_eur"].sum()),
        )
    _assert_partner_flows_from_sums(partner_sums)
    return pd.DataFrame(rows)


@pytask.task(id="aggregate_self_financing_heterogeneity")
def task_aggregate_self_financing_heterogeneity(
    path_to_aggregates: Annotated[Path, Product] = BLD
    / "derived"
    / "self_financing_heterogeneity_aggregates.pkl",
) -> None:
    """Build the long-format aggregator pickle.

    Reads the seven policy pickles at the canonical
    ``bld/solve_and_simulate/...`` paths. The baseline ledger is built
    first (it also defines the subgroup table and stays resident), then
    each counterfactual is loaded, aggregated, and freed one at a time via
    :func:`_build_aggregate_long_incremental`. This keeps peak memory at
    roughly two full ledgers (~3-4 GB) instead of holding all seven plus a
    copy (~20 GB), which previously triggered the OOM killer.
    """
    import gc

    inputs = load_specs()
    baseline_slim = load_slim(PATHS_BY_KEY["baseline"])
    sg = build_subgroup_table(baseline_slim, inputs.start_age)
    baseline_ledger = build_ledger(baseline_slim, inputs, "baseline", sg)
    del baseline_slim
    gc.collect()

    aggregate = _build_aggregate_long_incremental(inputs, sg, baseline_ledger)
    path_to_aggregates.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_pickle(path_to_aggregates)


# =============================================================================
# Writer
# =============================================================================


SUBGROUP_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("all", "all", "Total"),
    ("low", "all", "Low: Total"),
    ("low", "40-49", "Low: 40-49"),
    ("low", "50-59", "Low: 50-59"),
    ("low", "60-69", "Low: 60-69"),
    ("high", "all", "High: Total"),
    ("high", "40-49", "High: 40-49"),
    ("high", "50-59", "High: 50-59"),
    ("high", "60-69", "High: 60-69"),
)

MIN_CELL_N: int = 100

# Cells whose per-caregiver gross outlay G falls below this threshold print
# the SF percentage with a dagger: with a near-zero denominator the ratio
# explodes mechanically (e.g. Reform (PT) at ~394 EUR per caregiver) and
# should be read jointly with the absolute EUR surplus in the compact
# decomposition table.
MIN_OUTLAY_EUR: float = 500.0


def _delta(policy_row: pd.Series, baseline_row: pd.Series, col: str) -> float:
    """Return ``policy - baseline`` for column ``col``. NaN if either side
    is NaN. (Baseline row has the matched mask_source.)"""
    p = policy_row.get(col, np.nan)
    b = baseline_row.get(col, np.nan)
    if pd.isna(p) or pd.isna(b):
        return float("nan")
    return float(p) - float(b)


def _sf_components(policy_row: pd.Series, baseline_row: pd.Series) -> dict[str, float]:
    """Compute the SF building blocks from the matched policy and baseline
    rows, on a **ratio-of-totals** basis (masked EUR totals; the per-policy
    ever-caregiver counts never enter). The savings entries are
    ``baseline - policy`` (positive when the policy reduces gov outlay).

    Returned dict:

    - ``A``, ``dTaxIncome``, ``dPenSSC``, ``dUeSSC``, ``dLtcSSC``,
      ``dHealthSSC``: revenue deltas (policy - baseline) in EUR totals;
    - ``C_save``, ``D_save``, ``E_save``, ``F_save``: expenditure savings;
    - ``F_mech_save`` / ``F_behav_save``: split of ``F_save`` into the
      mechanical rule-change part (Pflegegeld abolished but would have been
      paid under the policy's care choices) and the behavioral part
      (induced change in Pflegegeld-eligible caregiving), via F-tilde;
    - ``G``: total gross leave outlay (denominator);
    - ``G_per_caregiver``: per-policy-caregiver average outlay (display /
      small-outlay flag only -- not used in the SF ratio);
    - ``n_base``: baseline ever-caregivers in the cell (display scaling).
    """
    A = float(policy_row.get("A_total", 0.0)) - float(baseline_row.get("A_total", 0.0))
    if math.isnan(A):
        A = 0.0
    f_save = -_delta(policy_row, baseline_row, "F_total")
    f_behav_save = float(baseline_row.get("F_total", np.nan)) - float(
        policy_row.get("Ftilde_total", np.nan)
    )
    f_mech_save = float(policy_row.get("Ftilde_total", np.nan)) - float(
        policy_row.get("F_total", np.nan)
    )
    return {
        "A": A,
        "dTaxIncome": _delta(policy_row, baseline_row, "B_income_tax_total"),
        "dPenSSC": _delta(policy_row, baseline_row, "B_pension_ssc_total"),
        "dUeSSC": _delta(policy_row, baseline_row, "B_ue_ssc_total"),
        "dLtcSSC": _delta(policy_row, baseline_row, "B_ltc_ssc_total"),
        "dHealthSSC": _delta(policy_row, baseline_row, "B_health_ssc_total"),
        "C_save": -_delta(policy_row, baseline_row, "C_total"),
        "D_save": -_delta(policy_row, baseline_row, "D_total"),
        "E_save": -_delta(policy_row, baseline_row, "E_total"),
        "F_save": f_save,
        "F_mech_save": f_mech_save,
        "F_behav_save": f_behav_save,
        "G": float(policy_row.get("G_total", np.nan)),
        "G_per_caregiver": float(policy_row.get("G_avg", np.nan)),
        "n_base": float(baseline_row.get("n_ever_caregivers", np.nan)),
    }


def _b_lifecycle(d_tax_total: float, A: float, *, floor: bool = True) -> float:
    """B = d_total_tax - A, floored at 0 only when ``floor=True``.

    The floor is kept for the back-comparability configs (iv) and (v); the
    honest configs (i) and (iii) report negative lifecycle tax effects
    instead of censoring them.
    """
    if math.isnan(d_tax_total) or math.isnan(A):
        return float("nan")
    b = d_tax_total - A
    return max(b, 0.0) if floor else b


def _b_lifecycle_no_pension(
    d_tax_total: float, d_pen: float, A: float, *, floor: bool = True
) -> float:
    """B* = lifecycle tax surplus excluding the pension-SSC piece."""
    if math.isnan(d_tax_total) or math.isnan(d_pen) or math.isnan(A):
        return float("nan")
    b = (d_tax_total - d_pen) - A
    return max(b, 0.0) if floor else b


def _sf_config_i(c: dict[str, float]) -> float:
    """MVPF-honest, SSC scope S4 (LTC-symmetric). No B floor.

    All modelled branches enter with both sides: pension (SSC in, payouts
    out via E), UE (SSC in, transfers out via C), LTC (SSC in,  # codespell:ignore
    Pflegegeld and formal care out via F and D). Health SSC excluded (no modelled
    health expenditure).
    """
    if math.isnan(c["G"]) or c["G"] <= 0:
        return float("nan")
    pieces = [c["dTaxIncome"], c["dPenSSC"], c["dUeSSC"], c["dLtcSSC"]]
    if any(math.isnan(p) for p in pieces):
        return float("nan")
    d_tax = sum(pieces)
    B = _b_lifecycle(d_tax, c["A"], floor=False)
    if math.isnan(B):
        return float("nan")
    num = c["A"] + B + c["C_save"] + c["D_save"] + c["E_save"] + c["F_save"]
    if math.isnan(num):
        return float("nan")
    return num / c["G"]


def _sf_config_iii(c: dict[str, float]) -> float:
    """Pension-excluded, SSC scope S4 (LTC-symmetric). No B floor.

    Both sides of the pension branch are dropped (no pension SSC in B*, no
    E in the numerator) -- the coherent alternative to Config (i).
    """
    if math.isnan(c["G"]) or c["G"] <= 0:
        return float("nan")
    pieces = [c["dTaxIncome"], c["dPenSSC"], c["dUeSSC"], c["dLtcSSC"]]
    if any(math.isnan(p) for p in pieces):
        return float("nan")
    d_tax = sum(pieces)
    B_star = _b_lifecycle_no_pension(d_tax, c["dPenSSC"], c["A"], floor=False)
    if math.isnan(B_star):
        return float("nan")
    num = c["A"] + B_star + c["C_save"] + c["D_save"] + c["F_save"]
    if math.isnan(num):
        return float("nan")
    return num / c["G"]


def _sf_config_iv(c: dict[str, float]) -> float:
    """Existing-published comparator, SSC scope S1 (broad incl. health).

    Same channel set as the published aggregate (A + B + C + D; no E, no
    F), with the B floor kept, but on the totals normalization and uniform
    masks. The contrast to Config (v) isolates the methodology delta.
    """
    if math.isnan(c["G"]) or c["G"] <= 0:
        return float("nan")
    pieces = [
        c["dTaxIncome"],
        c["dPenSSC"],
        c["dUeSSC"],
        c["dLtcSSC"],
        c["dHealthSSC"],
    ]
    if any(math.isnan(p) for p in pieces):
        return float("nan")
    d_tax = sum(pieces)
    B = _b_lifecycle(d_tax, c["A"], floor=True)
    if math.isnan(B):
        return float("nan")
    num = c["A"] + B + c["C_save"] + c["D_save"]
    if math.isnan(num):
        return float("nan")
    return num / c["G"]


def _sf_config_v(c: dict[str, float]) -> float:
    """Legacy hybrid back-comparability anchor, SSC scope S1 (broad).

    Same channel set as Config (iv) (A + B_S1 + C_save + D_save), no E,
    no F. The points of distinction are upstream: the components passed in
    must come from ``_sf_components_hybrid``, where (a) each policy
    (including baseline) is masked by its own caregiving rows and divided
    by its own ever-caregiver count, (b) the lifecycle tax delta is the
    direct model column ``total_tax_revenue`` (``dTaxTotal``), not the
    re-derived SSC decomposition, and (c) D is summed under the
    all-periods mask. This is the legacy ``task_fiscal.py`` /
    ``task_self_financing.py`` convention, used as the hard-fail
    back-comparability anchor (on the undiscounted cell; the legacy
    pipeline applies no discounting).
    """
    if math.isnan(c["G"]) or c["G"] <= 0:
        return float("nan")
    d_tax = c["dTaxTotal"]
    if math.isnan(d_tax):
        return float("nan")
    B = _b_lifecycle(d_tax, c["A"])
    if math.isnan(B):
        return float("nan")
    num = c["A"] + B + c["C_save"] + c["D_save"]
    if math.isnan(num):
        return float("nan")
    return num / c["G"]


CONFIG_FUNCTIONS: dict[str, callable] = {
    "i": _sf_config_i,
    "iii": _sf_config_iii,
    "iv": _sf_config_iv,
}


def _sf_components_hybrid(
    pol_row_active_cg: pd.Series,
    base_row_active_cg: pd.Series,
    pol_row_all_periods: pd.Series,
    base_row_all_periods: pd.Series,
) -> dict[str, float]:
    """Build SF building blocks under the legacy hybrid mask convention.

    A / B_* / C / G are pulled from rows aggregated under the
    active-caregiving mask (``perspective='active_caregiving'``); D is
    pulled from rows aggregated under the all-periods mask
    (``perspective='all_periods'``). Reproduces the channel-specific
    mask convention of the legacy ``task_self_financing.py`` aggregator.

    Output dict has the same shape as :func:`_sf_components` plus the key
    ``dTaxTotal``: the delta of the direct model column
    ``total_tax_revenue`` (per-caregiver averages), which replicates the
    legacy "Avg. total tax revenue" row exactly. E_save and F_save are
    populated from the active-CG rows (unused by Config (v) but kept for
    shape compatibility).
    """
    A = float(pol_row_active_cg.get("A_avg", 0.0)) - float(
        base_row_active_cg.get("A_avg", 0.0)
    )
    if math.isnan(A):
        A = 0.0
    return {
        "A": A,
        "dTaxTotal": _delta(pol_row_active_cg, base_row_active_cg, "T_total_avg"),
        "dTaxIncome": _delta(pol_row_active_cg, base_row_active_cg, "B_income_tax_avg"),
        "dPenSSC": _delta(pol_row_active_cg, base_row_active_cg, "B_pension_ssc_avg"),
        "dUeSSC": _delta(pol_row_active_cg, base_row_active_cg, "B_ue_ssc_avg"),
        "dLtcSSC": _delta(pol_row_active_cg, base_row_active_cg, "B_ltc_ssc_avg"),
        "dHealthSSC": _delta(pol_row_active_cg, base_row_active_cg, "B_health_ssc_avg"),
        "C_save": -_delta(pol_row_active_cg, base_row_active_cg, "C_avg"),
        "D_save": -_delta(pol_row_all_periods, base_row_all_periods, "D_avg"),
        "E_save": -_delta(pol_row_active_cg, base_row_active_cg, "E_avg"),
        "F_save": -_delta(pol_row_active_cg, base_row_active_cg, "F_avg"),
        "G": float(pol_row_active_cg.get("G_avg", np.nan)),
        "G_per_caregiver": float(pol_row_active_cg.get("G_avg", np.nan)),
        "n_base": float(base_row_active_cg.get("n_ever_caregivers", np.nan)),
    }


def _select_row(
    df: pd.DataFrame,
    *,
    policy: str,
    perspective: str,
    mask_source: str,
    mask_choice: str,
    discounting: str,
    education: str,
    age_bin: str,
) -> pd.Series | None:
    """Return the unique aggregator row matching the keys, or None."""
    sel = (
        (df["policy"] == policy)
        & (df["perspective"] == perspective)
        & (df["mask_source"] == mask_source)
        & (df["mask_choice"] == mask_choice)
        & (df["discounting"] == discounting)
        & (df["education"] == education)
        & (df["age_bin"] == age_bin)
    )
    sub = df[sel]
    if sub.empty:
        return None
    if len(sub) > 1:
        raise RuntimeError(
            f"Aggregator key not unique: policy={policy}, perspective={perspective}, "
            f"mask_source={mask_source}, mask_choice={mask_choice}, "
            f"discounting={discounting}, education={education}, age_bin={age_bin}"
        )
    return sub.iloc[0]


def _baseline_mask_source(
    perspective: str, counterfactual: str, mask_choice: str
) -> str:
    """Mask source under which the baseline aggregate is stored, for a
    given (perspective, counterfactual, mask_choice) comparison pair.

    Active caregiving always uses the counterfactual's mask (symmetric
    treatment-on-treated). Care demand uses the counterfactual's mask only
    under the intersection mask (Chapter 3 symmetric, Option B); under the
    headline mask it stays per-arm (``self``). All-periods is always ``self``.
    """
    if perspective == "active_caregiving":
        return counterfactual
    if perspective == "care_demand" and mask_choice == "intersection":
        return counterfactual
    return "self"


def _format_pct(value: float, *, dagger: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    suffix = r"$^{\dagger}$" if dagger else ""
    return f"{value * 100:.1f}\\%{suffix}"


@dataclass
class CellResult:
    """One table cell: SF degree, cell size, small-outlay flag, components."""

    sf: float
    n_cg: int
    small_outlay: bool = False
    components: dict | None = None


def _compute_cell(
    df: pd.DataFrame,
    *,
    config: str,
    policy: str,
    discounting: str,
    perspective: str,
    education: str,
    age_bin: str,
    mask_choice: str,
) -> CellResult:
    """Return the :class:`CellResult` for the single (config, policy, ...)
    cell. ``sf`` is NaN when the cell is masked out (no rows or below
    threshold).

    For ``config='v'`` (legacy hybrid back-comparability anchor) the
    rendered ``perspective`` argument is ignored: the hybrid mask is
    fixed-internal and pulls A/B/C/G from active-caregiving rows and D
    from all-periods rows. The three perspective rows in any (v)
    sub-panel therefore print identical numbers by construction.
    """
    if config == "v":
        return _compute_cell_hybrid(
            df,
            policy=policy,
            discounting=discounting,
            mask_choice=mask_choice,
            education=education,
            age_bin=age_bin,
        )
    baseline_ms = _baseline_mask_source(perspective, policy, mask_choice)
    pol_row = _select_row(
        df,
        policy=policy,
        perspective=perspective,
        mask_source=baseline_ms,
        mask_choice=mask_choice,
        discounting=discounting,
        education=education,
        age_bin=age_bin,
    )
    base_row = _select_row(
        df,
        policy="baseline",
        perspective=perspective,
        mask_source=baseline_ms,
        mask_choice=mask_choice,
        discounting=discounting,
        education=education,
        age_bin=age_bin,
    )
    if pol_row is None or base_row is None:
        return CellResult(sf=float("nan"), n_cg=0)
    n_cg = int(pol_row.get("n_ever_caregivers", 0))
    if n_cg < MIN_CELL_N:
        return CellResult(sf=float("nan"), n_cg=n_cg)
    components = _sf_components(pol_row, base_row)
    sf = CONFIG_FUNCTIONS[config](components)
    g_pc = components.get("G_per_caregiver", float("nan"))
    small = bool(not math.isnan(g_pc) and g_pc < MIN_OUTLAY_EUR)
    return CellResult(sf=sf, n_cg=n_cg, small_outlay=small, components=components)


def _compute_cell_hybrid(
    df: pd.DataFrame,
    *,
    policy: str,
    discounting: str,
    mask_choice: str,
    education: str,
    age_bin: str,
) -> CellResult:
    """Config (v) cell: legacy hybrid mask, perspective-invariant.

    Pulls A / dTaxTotal / C / G from rows with
    ``perspective='active_caregiving'`` where each policy (including the
    baseline) is masked by its OWN lagged-choice caregiving rows
    (``mask_source=policy`` for the policy ledger, ``mask_source='self'``
    for the baseline ledger) and divided by its own ever-caregiver count
    -- exactly the legacy ``task_fiscal.py`` convention behind the
    published master table. Pulls D from rows with
    ``perspective='all_periods'`` and ``mask_source='self'`` (the
    full-lifecycle mask). Produces a perspective-invariant SF degree (the
    rendered perspective row label is cosmetic).
    """
    pol_active = _select_row(
        df,
        policy=policy,
        perspective="active_caregiving",
        mask_source=policy,
        mask_choice=mask_choice,
        discounting=discounting,
        education=education,
        age_bin=age_bin,
    )
    base_active = _select_row(
        df,
        policy="baseline",
        perspective="active_caregiving",
        mask_source="self",
        mask_choice=mask_choice,
        discounting=discounting,
        education=education,
        age_bin=age_bin,
    )
    pol_all = _select_row(
        df,
        policy=policy,
        perspective="all_periods",
        mask_source="self",
        mask_choice=mask_choice,
        discounting=discounting,
        education=education,
        age_bin=age_bin,
    )
    base_all = _select_row(
        df,
        policy="baseline",
        perspective="all_periods",
        mask_source="self",
        mask_choice=mask_choice,
        discounting=discounting,
        education=education,
        age_bin=age_bin,
    )
    if any(r is None for r in (pol_active, base_active, pol_all, base_all)):
        return CellResult(sf=float("nan"), n_cg=0)
    n_cg = int(pol_active.get("n_ever_caregivers", 0))
    if n_cg < MIN_CELL_N:
        return CellResult(sf=float("nan"), n_cg=n_cg)
    components = _sf_components_hybrid(pol_active, base_active, pol_all, base_all)
    sf = _sf_config_v(components)
    g_pc = components.get("G_per_caregiver", float("nan"))
    small = bool(not math.isnan(g_pc) and g_pc < MIN_OUTLAY_EUR)
    return CellResult(sf=sf, n_cg=n_cg, small_outlay=small, components=components)


def _check_back_comparability_iv(df: pd.DataFrame, mask_choice: str = "v1") -> str:
    """Soft diagnostic on Config (iv): does the new uniform all-periods
    mask reproduce the legacy published 152.4 percent for Reform / total?

    Returns a one-line status message; never raises. Drift here is
    interpretable as a methodology delta (uniform all-periods mask vs the
    legacy hybrid mask), not a coding bug -- so the writer prints the
    line for inspection but does not abort.
    """
    sf = _compute_cell(
        df,
        config="iv",
        policy="full_beirat",
        discounting=LEGACY_DISCOUNTING,
        perspective="all_periods",
        education="all",
        age_bin="all",
        mask_choice=mask_choice,
    ).sf
    if math.isnan(sf):
        return (
            f"[SF heterogeneity {mask_choice}] Config (iv) back-comparability "
            "could not be evaluated (NaN cell)."
        )
    pct = sf * 100
    diff_pp = pct - EXISTING_REFORM_SF_PCT
    status = "OK" if abs(diff_pp) <= SANITY_TOLERANCE_PCT else "DRIFT"
    return (
        f"[SF heterogeneity {mask_choice}] Config (iv) discounted/all-periods/"
        f"Reform/total = {pct:.2f}% (existing published "
        f"{EXISTING_REFORM_SF_PCT:.1f}%, diff = {diff_pp:+.2f} pp -- {status}; "
        "soft warning, methodology delta from legacy hybrid mask -- not a bug)"
    )


def _check_back_comparability_v(
    df: pd.DataFrame, mask_choice: str = "v1"
) -> tuple[str, bool]:
    """Hard back-comparability anchor on Config (v): the legacy hybrid
    convention should reproduce the published 152.4 percent for Reform /
    total within ``SANITY_TOLERANCE_PCT`` percentage points.

    Evaluated on the UNDISCOUNTED cell: the legacy ``task_fiscal.py``
    pipeline sums raw EUR with no discounting, so that is the exact
    replication target (the discounted Config (v) panel is a variant, not
    the anchor).

    Returns (status_message, is_within_tolerance). The headline-mask
    writer raises ``RuntimeError`` if the check is out of tolerance; the
    intersection-mask writer prints the line but does not raise.
    """
    sf = _compute_cell(
        df,
        config="v",
        policy="full_beirat",
        discounting="undiscounted",
        perspective="all_periods",
        education="all",
        age_bin="all",
        mask_choice=mask_choice,
    ).sf
    if math.isnan(sf):
        return (
            f"[SF heterogeneity {mask_choice}] Config (v) back-comparability "
            "could not be evaluated (NaN cell).",
            False,
        )
    pct = sf * 100
    diff_pp = pct - EXISTING_REFORM_SF_PCT
    within = abs(diff_pp) <= SANITY_TOLERANCE_PCT
    status = "OK" if within else "FAIL"
    return (
        f"[SF heterogeneity {mask_choice}] Config (v) undiscounted/Reform/"
        f"total = {pct:.2f}% (existing published {EXISTING_REFORM_SF_PCT:.1f}%, "
        f"diff = {diff_pp:+.2f} pp -- {status}; hard anchor at "
        f"+/- {SANITY_TOLERANCE_PCT:.1f} pp)",
        within,
    )


# =============================================================================
# LaTeX rendering
# =============================================================================


def _format_panel_header(config: str, discounting: str) -> str:
    """Header line above the tabular block, identifying the panel.

    Rendered as a paragraph line (not a ``\\multicolumn`` row) so the panel
    body compiles standalone outside any enclosing tabular.
    """
    config_label = CONFIG_LABELS[config]
    disc_label = DISCOUNTING_LABELS.get(discounting, discounting)
    return (
        rf"\par\medskip\noindent\textbf{{{config_label} -- {disc_label}}}"
        r"\par\nopagebreak\smallskip"
    )


def _format_two_level_columns_header() -> tuple[str, str]:
    """Two-row header above the data: top row groups, bottom row labels."""
    top = (
        r"\multirow{2}{*}{Perspective} & "
        r"\multicolumn{1}{c}{} & "
        r"\multicolumn{4}{c}{Low education} & "
        r"\multicolumn{4}{c}{High education} \\"
    )
    cols = " & ".join(
        ["Total"]
        + ["Total", "40-49", "50-59", "60-69"]
        + ["Total", "40-49", "50-59", "60-69"]
    )
    bottom = rf" & {cols} \\"
    return top, bottom


def _format_data_row(
    df: pd.DataFrame,
    *,
    config: str,
    discounting: str,
    perspective: str,
    policy: str,
    mask_choice: str,
) -> str:
    """One LaTeX row per (perspective) within a reform sub-panel."""
    cells: list[str] = []
    for education, age_bin, _label in SUBGROUP_COLUMNS:
        res = _compute_cell(
            df,
            config=config,
            policy=policy,
            discounting=discounting,
            perspective=perspective,
            education=education,
            age_bin=age_bin,
            mask_choice=mask_choice,
        )
        cells.append(_format_pct(res.sf, dagger=res.small_outlay))
    label = PERSPECTIVE_LABELS[perspective]
    return f"    {label} & " + " & ".join(cells) + r" \\"


def _format_reform_subpanel(
    df: pd.DataFrame,
    *,
    config: str,
    discounting: str,
    policy: str,
    sub_panel_letter: str,
    mask_choice: str,
) -> list[str]:
    """A reform-specific sub-panel: title row + 3 perspective rows."""
    label = POLICY_LABELS_SHORT[policy]
    out = [
        (
            rf"    \multicolumn{{10}}{{l}}"
            rf"{{\textit{{Sub-panel {sub_panel_letter} -- {label}}}}}"
            r" \\"
        ),
        r"    \midrule",
    ]
    for perspective in PERSPECTIVES:
        out.append(
            _format_data_row(
                df,
                config=config,
                discounting=discounting,
                perspective=perspective,
                policy=policy,
                mask_choice=mask_choice,
            )
        )
    out.append(r"    \addlinespace")
    return out


def _format_panel(
    df: pd.DataFrame,
    *,
    config: str,
    discounting: str,
    mask_choice: str,
) -> list[str]:
    """A full panel: 1 header + 5 reform sub-panels (each 3 perspective rows)."""
    out: list[str] = []
    out.append(_format_panel_header(config, discounting))
    out.extend(
        (
            r"\begin{tabular}{l r r r r r r r r r}",
            r"\toprule",
        )
    )
    top, bottom = _format_two_level_columns_header()
    out.extend((top, bottom, r"\midrule"))
    sub_letters = "ABCDEF"
    for letter, policy in zip(sub_letters, COUNTERFACTUALS, strict=True):
        out.extend(
            _format_reform_subpanel(
                df,
                config=config,
                discounting=discounting,
                policy=policy,
                sub_panel_letter=letter,
                mask_choice=mask_choice,
            )
        )
    out.extend((r"\bottomrule", r"\end{tabular}", ""))
    return out


# --- Notes blocks -----------------------------------------------------------

MASK_CAPTION: dict[str, str] = {
    "v1": (
        "Headline mask: care-demand periods use each policy's own "
        r"\(\mathrm{care\_demand} > 0\) mask; active-caregiving periods "
        "use each counterfactual's lagged-choice mask, applied "
        "symmetrically to the baseline; all-periods uses no row mask."
    ),
    "intersection": (
        r"Robustness mask. Care-demand row: \emph{symmetric} intersection -- "
        r"both the policy and baseline columns are restricted to "
        r"\(\mathrm{policy\_care\_demand}>0 \wedge "
        r"\mathrm{baseline\_care\_demand}>0\) (contemporaneous). "
        r"Active-caregiving row: each counterfactual's lagged-choice mask "
        r"(applied symmetrically) AND the \emph{lagged} baseline care demand, "
        r"to match the lagged-choice definition. All-periods row: \emph{not} "
        r"restricted -- survival is verified identical across arms, so the "
        r"survival-aligned support is the full lifecycle. The care-demand/active "
        r"restrictions drop only the small (\(\sim\)3.7\%) set of cells where "
        r"care demand differs across arms (care demand is meant to be "
        r"policy-invariant; the residual mismatch is care-demand non-invariance "
        r"\textemdash{} likely a simulation RNG artefact, not mortality timing)."
    ),
}

_NOTES_CHANNELS = (
    r"\textit{Channels (all in EUR):} "
    r"$A$ = within-spell tax/SSC delta on the wage-replacement top-up "
    r"(policy $-$ baseline); "
    r"$\Delta T$ = lifecycle tax-and-SSC revenue delta under the stated SSC "
    r"scope; $B$ = $\Delta T - A$ (lifecycle surplus net of the within-spell "
    r"part); "
    r"$C$ = saved unemployment transfers (baseline $-$ policy); "
    r"$D$ = saved imputed formal-care costs; "
    r"$E$ = saved public pension payouts (own + partner); "
    r"$F$ = saved Pflegegeld (care allowance) outlays; "
    r"$G$ = gross wage-replacement outlay (denominator)."
)

_NOTES_SCOPES = (
    r"\textit{SSC scopes:} S4 (LTC-symmetric) = income tax + pension SSC + "
    r"UE SSC + LTC SSC; health SSC is excluded because the "  # codespell:ignore
    r"model has no "
    r"health expenditure margin (conservative; see the compact decomposition "
    r"table for the excluded $\Delta$health-SSC memo). "
    r"S1 (broad) = S4 + health SSC, matching the published comparator."
)

_NOTES_COMMON = (
    r"\textit{Normalization:} configs (i), (iii), (iv) compute the SF degree "
    r"as a ratio of masked EUR totals (policy $-$ baseline), which is "
    r"invariant to policy-induced changes in the number of caregivers; "
    r"config (v) keeps the legacy per-policy-caregiver averages for "
    r"back-comparability. "
    r"\textit{Perspectives:} care-demand periods (parent requires care), "
    r"active-caregiving periods (agent provides informal care, lagged "
    r"choice), all periods (full lifecycle). "
    r"\textit{Subgroups:} education and age at first parental care demand, "
    r"both fixed from the baseline simulation; agents whose parent never "
    r"requires care appear only in the Total column. "
    r"\textit{Discounting:} the discounted panel uses the primary government "
    r"discount rate of 2.25\% (the model's own \texttt{interest\_rate}); the "
    r"1\% and 3\% sweep variants are in the compact discount-rate sensitivity "
    r"table. Per-period weight \((1+r)^{-\text{period}}\). "
    rf"\textit{{Flags:}} cells with fewer than {MIN_CELL_N} ever-caregivers "
    r"print ``--''; $^{\dagger}$ marks cells whose per-caregiver gross "
    rf"outlay $G$ is below EUR {MIN_OUTLAY_EUR:.0f}, where the SF ratio is "
    r"mechanically inflated by a near-zero denominator and should be read "
    r"jointly with the absolute EUR rows of the compact decomposition table."
)

_NOTES_NEGATIVE_SF = (
    r"\textit{Reading the sign:} the degree is induced fiscal gains divided by "
    r"the gross outlay $G > 0$, so a \emph{negative} degree means the policy's "
    r"induced responses cost the government money over and above the outlay "
    r"itself (total fiscal cost $G\,(1-\mathrm{SF})$), while $\mathrm{SF} > "
    r"100\%$ means it more than pays for itself. A degree turns negative when the "
    r"within-window fiscal losses outweigh the savings: income tax and SSC "
    r"revenue fall while the parent needs care (reduced labour supply), and in "
    r"the S4 headline (config i) the policy additionally raises public-pension "
    r"payouts ($E$) and Pflegegeld ($F$); the saved formal-care cost ($D$, the "
    r"baseline formal care that the policy converts to informal care) is large "
    r"and positive in \emph{every} perspective and is what offsets these losses. "
    r"Which perspective is most negative is not universal: the care-demand and "
    r"active-caregiving windows capture the within-spell revenue loss but exclude "
    r"the post-spell tax recovery from preserved employment, so they are usually "
    r"less favourable than the all-periods row, which nets the two and is the "
    r"coherent lifecycle statement. For $^{\dagger}$ cells (near-zero $G$) the "
    r"sign is meaningful but the magnitude is not."
)

CONFIG_FORMULA_NOTES: dict[str, str] = {
    "i": (
        r"\textit{Formula (Config i, headline):} "
        r"$\mathrm{SF} = (A + B + C + D + E + F)\, /\, G$ with "
        r"$B = \Delta T_{S4} - A$ (no floor; negative lifecycle tax effects "
        r"are reported, not censored). All modelled insurance branches "
        r"enter with both sides: pension (SSC in, payouts out via $E$), "
        r"unemployment (SSC in, transfers out via $C$), long-term care "
        r"(SSC in, Pflegegeld and formal care out via $F$ and $D$)."
    ),
    "iii": (
        r"\textit{Formula (Config iii):} "
        r"$\mathrm{SF} = (A + B^{*} + C + D + F)\, /\, G$ with "
        r"$B^{*} = \Delta T_{S4} - \Delta\mathrm{PensionSSC} - A$ (no "
        r"floor). The pension branch is excluded on both sides (no pension "
        r"SSC in $B^{*}$, no $E$): the coherent alternative to Config (i)."
    ),
    "iv": (
        r"\textit{Formula (Config iv):} "
        r"$\mathrm{SF} = (A + B + C + D)\, /\, G$ with "
        r"$B = \max(\Delta T_{S1} - A,\, 0)$ (floor kept). Same channel set "
        r"as the published aggregate ($A+B+C+D$; no $E$, no $F$) but on the "
        r"totals normalization and uniform masks; the contrast to Config "
        r"(v) isolates the methodology delta."
    ),
    "v": (
        r"\textit{Formula (Config v, legacy anchor):} "
        r"$\mathrm{SF} = (A + B + C + D)\, /\, G$ with "
        r"$B = \max(\Delta T - A,\, 0)$, where $\Delta T$ is the direct "
        r"model total-tax-revenue column (income tax + own SSC + partner "
        r"SSC, i.e.\ the broad S1 scope), computed on per-policy-caregiver "
        r"averages under the legacy hybrid mask: $A$, $\Delta T$, $C$, $G$ "
        r"from each policy's OWN active-caregiving rows (baseline masked "
        r"by its own caregiving rows, not symmetrically), $D$ from "
        r"all-periods rows. The undiscounted panel reproduces the "
        rf"published {EXISTING_REFORM_SF_PCT:.1f}\% for the Reform (hard "
        r"back-comparability anchor); perspective rows are identical by "
        r"construction."
    ),
}


def _format_notes_block(config: str, mask_choice: str) -> list[str]:
    """Footnote minipage appended below the tabular blocks of one file."""
    return [
        r"\vspace{0.5em}",
        r"\begin{minipage}{\textwidth}",
        r"\scriptsize",
        r"\textbf{Notes.} " + MASK_CAPTION[mask_choice],
        "",
        CONFIG_FORMULA_NOTES[config],
        "",
        _NOTES_CHANNELS,
        "",
        _NOTES_SCOPES,
        "",
        _NOTES_COMMON,
        "",
        _NOTES_NEGATIVE_SF,
        r"\end{minipage}",
    ]


def _format_config_table(df: pd.DataFrame, *, config: str, mask_choice: str) -> str:
    """One .tex file: a single config, one table float per discounting panel.

    To keep the compendium readable, the per-config files show the PRIMARY
    (2.25%, model r) discounted panel and the undiscounted panel; the other
    sweep rates (1%, 3%) live in the aggregator and the compact discount-rate
    sensitivity table. Panels are separate ``table`` environments separated by
    ``\\clearpage`` so each fits on one PDF page.
    """
    mask_label = "headline mask" if mask_choice == "v1" else "intersection mask"
    panel_discountings = (PRIMARY_DISCOUNTING, "undiscounted")
    lines: list[str] = []
    lines.extend(
        (
            r"% Auto-generated by task_self_financing_heterogeneity.py.",
            r"% Do not edit by hand.",
            "",
        )
    )
    for idx, discounting in enumerate(panel_discountings):
        disc_label = DISCOUNTING_LABELS.get(discounting, discounting)
        disc_suffix = discounting
        lines.extend((r"\begin{table}[htbp]", r"  \centering"))
        lines.append(
            rf"  \caption{{Self-financing heterogeneity -- {CONFIG_LABELS[config]} "
            rf"({mask_label}, {disc_label}).}}"
        )
        lines.extend(
            (
                rf"  \label{{tab:sf_heterogeneity_config_{config}_"
                rf"{mask_choice}_{disc_suffix}}}",
                r"  \scriptsize",
            )
        )
        lines.extend(
            _format_panel(
                df,
                config=config,
                discounting=discounting,
                mask_choice=mask_choice,
            )
        )
        lines.extend(_format_notes_block(config, mask_choice))
        lines.extend((r"\end{table}", ""))
        if idx < len(panel_discountings) - 1:
            lines.extend((r"\clearpage", ""))
    return "\n".join(lines)


def _write_config_tables(
    df: pd.DataFrame,
    *,
    paths_by_config: dict[str, Path],
    mask_choice: str,
    enforce_config_v_anchor: bool,
) -> None:
    """Write one LaTeX file per config and run the back-comparability checks.

    The Config (iv) check is always a soft print. The Config (v) check is
    always printed; when ``enforce_config_v_anchor=True`` (the headline
    writer) a drift outside ``SANITY_TOLERANCE_PCT`` raises
    ``RuntimeError`` after all files are written. The intersection-mask
    writer prints but does not enforce.
    """
    for config, path in paths_by_config.items():
        body = _format_config_table(df, config=config, mask_choice=mask_choice)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    print(_check_back_comparability_iv(df, mask_choice=mask_choice), flush=True)
    msg_v, within_v = _check_back_comparability_v(df, mask_choice=mask_choice)
    print(msg_v, flush=True)
    if enforce_config_v_anchor and not within_v:
        raise RuntimeError(
            "Config (v) legacy-hybrid back-comparability anchor failed: "
            f"{msg_v}. The legacy hybrid mask should reproduce the existing "
            f"published 152.4% within +/- {SANITY_TOLERANCE_PCT:.1f} pp. "
            "Investigate before publishing."
        )


SF_HET_DIR: Path = BLD / "tables" / "publication" / "self_financing_heterogeneity"

HET_TEX_PATHS_HEADLINE: dict[str, Path] = {
    config: SF_HET_DIR / f"sf_heterogeneity_config_{config}.tex"
    for config in CONFIG_ORDER
}
HET_TEX_PATHS_INTERSECTION: dict[str, Path] = {
    config: SF_HET_DIR / f"sf_heterogeneity_config_{config}_intersection.tex"
    for config in CONFIG_ORDER
}


@pytask.task(id="write_self_financing_heterogeneity_tables")
def task_write_self_financing_heterogeneity_tables(
    path_to_aggregates: Path = BLD
    / "derived"
    / "self_financing_heterogeneity_aggregates.pkl",
    paths_to_tex: Annotated[dict[str, Path], Product] = HET_TEX_PATHS_HEADLINE,
) -> None:
    """Emit the mask_v1 headline LaTeX tables (one file per config; each file
    has discounted and undiscounted panels in separate table floats).
    Hard-fails on Config (v) back-comparability drift."""
    df = pd.read_pickle(path_to_aggregates)
    _write_config_tables(
        df,
        paths_by_config=paths_to_tex,
        mask_choice="v1",
        enforce_config_v_anchor=True,
    )


@pytask.task(id="write_self_financing_heterogeneity_tables_intersection_mask")
def task_write_self_financing_heterogeneity_tables_intersection_mask(
    path_to_aggregates: Path = BLD
    / "derived"
    / "self_financing_heterogeneity_aggregates.pkl",
    paths_to_tex: Annotated[dict[str, Path], Product] = HET_TEX_PATHS_INTERSECTION,
) -> None:
    """Emit the intersection-mask robustness LaTeX tables (one file per
    config). Prints both back-comparability checks but does not enforce."""
    df = pd.read_pickle(path_to_aggregates)
    _write_config_tables(
        df,
        paths_by_config=paths_to_tex,
        mask_choice="intersection",
        enforce_config_v_anchor=False,
    )


# =============================================================================
# Compact tables (Phase 4): headline degrees + Config (i) decomposition
# =============================================================================


def _format_eur(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:,.0f}"


def _compact_degrees_body(df: pd.DataFrame) -> str:
    """Compact degrees table: rows = config x perspective, columns = the six
    counterfactual policies. Total subgroup, discounted, headline mask."""
    n_pol = len(COUNTERFACTUALS)
    lines: list[str] = []
    lines.extend(
        (
            r"% Auto-generated by task_self_financing_heterogeneity.py.",
            r"% Do not edit by hand.",
            "",
            r"\begin{table}[htbp]",
            r"  \centering",
        )
    )
    lines.extend(
        (
            r"  \caption{Self-financing degrees -- compact overview "
            r"(Total population, discounted at 2.25\% [model $r$], headline mask).}",
            r"  \label{tab:sf_compact_degrees}",
            r"  \small",
            r"  \begin{tabular}{ll" + "r" * n_pol + "}",
            r"  \toprule",
        )
    )
    header = " & ".join(POLICY_LABELS_SHORT[p] for p in COUNTERFACTUALS)
    lines.extend(
        (
            rf"  Configuration & Perspective & {header} \\",
            r"  \midrule",
        )
    )
    for config in CONFIG_ORDER:
        perspectives = PERSPECTIVES if config != "v" else ("all_periods",)
        for k, perspective in enumerate(perspectives):
            cells = []
            for policy in COUNTERFACTUALS:
                res = _compute_cell(
                    df,
                    config=config,
                    policy=policy,
                    discounting=PRIMARY_DISCOUNTING,
                    perspective=perspective,
                    education="all",
                    age_bin="all",
                    mask_choice="v1",
                )
                cells.append(_format_pct(res.sf, dagger=res.small_outlay))
            config_cell = (
                rf"\multirow{{{len(perspectives)}}}{{*}}{{({config})}}"
                if k == 0
                else ""
            )
            persp_label = (
                PERSPECTIVE_LABELS[perspective]
                if config != "v"
                else "Legacy hybrid mask"
            )
            lines.append(
                f"  {config_cell} & {persp_label} & " + " & ".join(cells) + r" \\"
            )
        lines.append(r"  \addlinespace")
    lines.extend(
        (
            r"  \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.5em}",
            r"  \begin{minipage}{\textwidth}",
            r"  \scriptsize",
        )
    )
    lines.extend(
        (
            r"  \textbf{Notes.} Self-financing degree = induced fiscal gains / "
            r"gross wage-replacement outlay, Total population, discounted, "
            r"headline mask. Configurations: (i) MVPF-honest, S4, no $B$ floor "
            r"(headline); (iii) pension branch excluded, S4, no floor; (iv) "
            r"published channel set $A+B+C+D$, S1, floor kept, totals "
            r"normalization; (v) legacy hybrid anchor on per-caregiver averages "
            rf"(reproduces {EXISTING_REFORM_SF_PCT:.1f}\% for the Reform). "
            r"Configs (i), (iii), (iv) are ratios of masked EUR totals. "
            rf"$^{{\dagger}}$ = per-caregiver outlay below EUR "
            rf"{MIN_OUTLAY_EUR:.0f} (denominator mechanically small; see the "
            r"compact decomposition table for absolute EUR magnitudes). "
            r"See the per-config heterogeneity tables for full formulas and "
            r"subgroup splits.",
            "",
            r"  " + _NOTES_NEGATIVE_SF,
            r"  \end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


def _compact_decomposition_body(df: pd.DataFrame) -> str:
    """Compact decomposition table for Config (i), all-periods perspective,
    discounted, headline mask, Total subgroup. Channel rows in EUR per
    baseline ever-caregiver; memo rows for the F split and the excluded
    health SSC; standing EUR rows; final SF row."""
    results: dict[str, CellResult] = {}
    for policy in COUNTERFACTUALS:
        results[policy] = _compute_cell(
            df,
            config="i",
            policy=policy,
            discounting=PRIMARY_DISCOUNTING,
            perspective="all_periods",
            education="all",
            age_bin="all",
            mask_choice="v1",
        )

    def per_cg(policy: str, fn) -> float:
        res = results[policy]
        if res.components is None:
            return float("nan")
        n_base = res.components.get("n_base", float("nan"))
        if math.isnan(n_base) or n_base <= 0:
            return float("nan")
        return fn(res.components) / n_base

    def b_no_floor(c: dict[str, float]) -> float:
        d_tax = c["dTaxIncome"] + c["dPenSSC"] + c["dUeSSC"] + c["dLtcSSC"]
        return _b_lifecycle(d_tax, c["A"], floor=False)

    def net_surplus(c: dict[str, float]) -> float:
        return (
            c["A"]
            + b_no_floor(c)
            + c["C_save"]
            + c["D_save"]
            + c["E_save"]
            + c["F_save"]
            - c["G"]
        )

    channel_rows: list[tuple[str, callable]] = [
        (r"$A$: within-spell tax/SSC on top-up", operator.itemgetter("A")),
        (r"$B$: lifecycle tax surplus (S4, no floor)", b_no_floor),
        (r"$C$: saved UE transfers", operator.itemgetter("C_save")),  # codespell:ignore
        (r"$D$: saved formal-care costs", operator.itemgetter("D_save")),
        (r"$E$: saved pension payouts", operator.itemgetter("E_save")),
        (r"$F$: saved Pflegegeld", operator.itemgetter("F_save")),
        (
            r"\quad\textit{memo: $F$ mechanical (rule change)}",
            operator.itemgetter("F_mech_save"),
        ),
        (
            r"\quad\textit{memo: $F$ behavioral (induced take-up)}",
            operator.itemgetter("F_behav_save"),
        ),
        (
            r"\quad\textit{memo: $\Delta$health SSC (excluded from S4)}",
            operator.itemgetter("dHealthSSC"),
        ),
    ]
    eur_rows: list[tuple[str, callable]] = [
        (r"Net fiscal surplus", net_surplus),
        (r"Gross outlay $G$", operator.itemgetter("G")),
    ]

    n_pol = len(COUNTERFACTUALS)
    lines: list[str] = []
    lines.extend(
        (
            r"% Auto-generated by task_self_financing_heterogeneity.py.",
            r"% Do not edit by hand.",
            "",
            r"\begin{table}[htbp]",
            r"  \centering",
        )
    )
    lines.append(
        r"  \caption{Fiscal decomposition of the headline self-financing "
        r"degree -- Config (i), all periods, discounted at 2.25\% (model $r$), "
        r"headline mask (EUR per baseline ever-caregiver).}"
    )
    lines.extend(
        (
            r"  \label{tab:sf_compact_decomposition}",
            r"  \small",
            r"  \begin{tabular}{l" + "r" * n_pol + "}",
            r"  \toprule",
        )
    )
    header = " & ".join(POLICY_LABELS_SHORT[p] for p in COUNTERFACTUALS)
    lines.extend(
        (
            rf"  Channel & {header} \\",
            r"  \midrule",
        )
    )
    for label, fn in channel_rows:
        cells = [_format_eur(per_cg(p, fn)) for p in COUNTERFACTUALS]
        lines.append(f"  {label} & " + " & ".join(cells) + r" \\")
    lines.append(r"  \midrule")
    for label, fn in eur_rows:
        cells = [_format_eur(per_cg(p, fn)) for p in COUNTERFACTUALS]
        lines.append(f"  {label} & " + " & ".join(cells) + r" \\")
    lines.append(r"  \midrule")
    sf_cells = [
        _format_pct(results[p].sf, dagger=results[p].small_outlay)
        for p in COUNTERFACTUALS
    ]
    lines.append(r"  Self-financing degree & " + " & ".join(sf_cells) + r" \\")
    lines.extend(
        (
            r"  \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.5em}",
            r"  \begin{minipage}{\textwidth}",
            r"  \scriptsize",
        )
    )
    lines.extend(
        (
            r"  \textbf{Notes.} All rows except the last are masked EUR totals "
            r"(policy $-$ baseline) divided by the number of baseline ever-"
            r"caregivers; the SF degree itself is the ratio of totals "
            r"$(A+B+C+D+E+F)/G$ and does not depend on this display scaling. "
            r"The $F$ memo rows split the Pflegegeld saving into the mechanical "
            r"part (abolition of payments that the policy's own care choices "
            r"would have triggered, via imputed $\tilde F$) and the behavioral "
            r"part (induced change in Pflegegeld-eligible caregiving); they sum "
            r"to the $F$ row. The $\Delta$health-SSC memo row reports the "
            r"revenue channel excluded from the S4 scope (it would be added to "
            r"$B$ under the broad S1 scope). Net fiscal surplus = "
            r"$A+B+C+D+E+F-G$. "
            rf"$^{{\dagger}}$ = per-caregiver outlay below EUR {MIN_OUTLAY_EUR:.0f}.",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


SWEEP_DISCOUNTINGS: tuple[str, ...] = ("undiscounted", "disc1", "disc2p25", "disc3")
SWEEP_RATE_HEADERS: dict[str, str] = {
    "undiscounted": "Undisc.",
    "disc1": "1\\%",
    "disc2p25": "2.25\\%",
    "disc3": "3\\%",
}


def _compact_discount_sensitivity_body(df: pd.DataFrame) -> str:
    """Discount-rate sensitivity of the headline degree.

    Config (i), all-periods, Total, headline mask. Rows = the six reform
    variants; columns = the discount-rate sweep {undiscounted, 1, 2.25, 3}%.
    Shows how the headline self-financing degree moves with the government
    discount rate (2.25% = model r is the primary headline).
    """
    n_rate = len(SWEEP_DISCOUNTINGS)
    lines: list[str] = []
    lines.extend(
        (
            r"% Auto-generated by task_self_financing_heterogeneity.py.",
            r"% Do not edit by hand.",
            "",
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Discount-rate sensitivity of the headline "
            r"self-financing degree -- Config (i) MVPF-honest, all periods, "
            r"Total population, headline mask. Primary = 2.25\% (model $r$).}",
            r"  \label{tab:sf_discount_sensitivity}",
            r"  \small",
            r"  \begin{tabular}{l" + "r" * n_rate + "}",
            r"  \toprule",
        )
    )
    header = " & ".join(SWEEP_RATE_HEADERS[d] for d in SWEEP_DISCOUNTINGS)
    lines.extend((rf"  Reform variant & {header} \\", r"  \midrule"))
    for policy in COUNTERFACTUALS:
        cells = []
        for disc in SWEEP_DISCOUNTINGS:
            res = _compute_cell(
                df,
                config="i",
                policy=policy,
                discounting=disc,
                perspective="all_periods",
                education="all",
                age_bin="all",
                mask_choice="v1",
            )
            cells.append(_format_pct(res.sf, dagger=res.small_outlay))
        lines.append(f"  {POLICY_LABELS_SHORT[policy]} & " + " & ".join(cells) + r" \\")
    lines.extend(
        (
            r"  \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.5em}",
            r"  \begin{minipage}{\textwidth}",
            r"  \scriptsize",
            r"  \textbf{Notes.} Government discount rate swept over "
            r"$\{0, 1, 2.25, 3\}\%$ real (per-period weight "
            r"$(1+r)^{-\text{period}}$). 2.25\% equals the model's own "
            r"\texttt{interest\_rate} and is the internally-consistent primary "
            r"headline; 1\% is the legacy Stuermer-Heiber convention; 3\% a "
            r"conventional social discount rate. Raising the rate lifts Config "
            r"(i) (it down-weights the far-future pension-payout cost $E$) and "
            r"lowers the pension-excluded configs -- the degree's sensitivity to "
            r"the rate and to the pension-channel treatment is the honest result. "
            rf"$^{{\dagger}}$ = per-caregiver outlay below EUR {MIN_OUTLAY_EUR:.0f}.",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


@pytask.task(id="write_self_financing_compact_tables")
def task_write_self_financing_compact_tables(
    path_to_aggregates: Path = BLD
    / "derived"
    / "self_financing_heterogeneity_aggregates.pkl",
    path_to_degrees_tex: Annotated[Path, Product] = SF_HET_DIR
    / "sf_compact_degrees.tex",
    path_to_decomposition_tex: Annotated[Path, Product] = SF_HET_DIR
    / "sf_compact_decomposition.tex",
    path_to_discount_sensitivity_tex: Annotated[Path, Product] = SF_HET_DIR
    / "sf_discount_sensitivity.tex",
) -> None:
    """Emit the compact lead tables (degrees overview + Config (i)
    decomposition with memo and EUR rows + discount-rate sensitivity)."""
    df = pd.read_pickle(path_to_aggregates)
    path_to_degrees_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_degrees_tex.write_text(_compact_degrees_body(df), encoding="utf-8")
    path_to_decomposition_tex.write_text(
        _compact_decomposition_body(df), encoding="utf-8"
    )
    path_to_discount_sensitivity_tex.write_text(
        _compact_discount_sensitivity_body(df), encoding="utf-8"
    )


# =============================================================================
# PDF report (Phase 5): one compiled document with all SF tables
# =============================================================================

SF_REPORT_PREAMBLE = r"""% Auto-generated by task_self_financing_heterogeneity.py.
% Do not edit.
\documentclass[10pt]{article}
\usepackage[a4paper, landscape, margin=1.5cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{caption}
\usepackage{float}
% Legacy tables reference \citet; resolve without a bibliography.
\providecommand{\citet}[1]{St\"urmer and Heiber (2024)}
\providecommand{\citep}[1]{(St\"urmer and Heiber, 2024)}
\title{Self-Financing of Caregiving-Leave Policies\\
\large Complete table compendium (auto-generated)}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


def _report_section(title: str, tex_paths: list[Path]) -> str:
    parts = [rf"\section{{{title}}}"]
    for p in tex_paths:
        body = p.read_text(encoding="utf-8")
        # Float placement: force each table onto its own page region in the
        # report to keep landscape pages readable.
        body = body.replace(r"\begin{table}[htbp]", r"\begin{table}[H]")
        parts.extend((body, r"\clearpage"))
    return "\n".join(parts)


@pytask.task(id="compile_self_financing_report")
def task_compile_self_financing_report(
    path_to_degrees_tex: Path = SF_HET_DIR / "sf_compact_degrees.tex",
    path_to_decomposition_tex: Path = SF_HET_DIR / "sf_compact_decomposition.tex",
    path_to_discount_sensitivity_tex: Path = SF_HET_DIR / "sf_discount_sensitivity.tex",
    paths_to_het_headline: dict[str, Path] = HET_TEX_PATHS_HEADLINE,
    paths_to_het_intersection: dict[str, Path] = HET_TEX_PATHS_INTERSECTION,
    path_to_legacy_degree: Path = BLD
    / "tables"
    / "publication"
    / "self_financing_degree.tex",
    path_to_legacy_decomposition: Path = BLD
    / "tables"
    / "publication"
    / "self_financing_decomposition.tex",
    path_to_report_tex: Annotated[Path, Product] = SF_HET_DIR / "sf_report.tex",
    path_to_report_pdf: Annotated[Path, Product] = SF_HET_DIR / "sf_report.pdf",
) -> None:
    """Assemble sf_report.tex (compact lead tables, per-config heterogeneity
    tables for both masks, legacy Stuermer-Heiber tables) and compile it
    with latexmk."""
    import subprocess

    sections: list[str] = []
    sections.extend(
        (
            _report_section(
                "Compact overview (lead tables)",
                [
                    path_to_degrees_tex,
                    path_to_decomposition_tex,
                    path_to_discount_sensitivity_tex,
                ],
            ),
        )
    )
    sections.extend(
        (
            _report_section(
                "Heterogeneity tables -- headline mask",
                [paths_to_het_headline[c] for c in CONFIG_ORDER],
            ),
            _report_section(
                "Heterogeneity tables -- intersection mask (robustness)",
                [paths_to_het_intersection[c] for c in CONFIG_ORDER],
            ),
        )
    )
    legacy = [
        p for p in (path_to_legacy_degree, path_to_legacy_decomposition) if p.exists()
    ]
    if legacy:
        sections.append(
            _report_section("Legacy aggregate tables (published comparator)", legacy)
        )
    else:
        print(
            "[sf_report] WARNING: legacy self-financing tables not found; "
            "report compiled without them.",
            flush=True,
        )

    report = SF_REPORT_PREAMBLE + "\n".join(sections) + "\n\\end{document}\n"
    path_to_report_tex.parent.mkdir(parents=True, exist_ok=True)
    path_to_report_tex.write_text(report, encoding="utf-8")

    result = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={path_to_report_tex.parent}",
            str(path_to_report_tex),
        ],
        capture_output=True,
        text=True,
        cwd=path_to_report_tex.parent,
        check=False,
    )
    if result.returncode != 0 or not path_to_report_pdf.exists():
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"latexmk failed (exit {result.returncode}). Tail of log:\n{tail}"
        )
    print(f"[sf_report] compiled {path_to_report_pdf}", flush=True)
