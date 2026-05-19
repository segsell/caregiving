"""Summary statistics tasks for publication (e.g. structural estimation sample)."""

import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    FULL_TIME_CHOICES,
    GOOD_HEALTH,
    PART_TIME_CHOICES,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED_CHOICES,
)
from caregiving.moments.task_create_soep_moments import create_df_with_caregivers

# Scalars from shared choice codes (structural sample uses same codes)
FULL_TIME = int(FULL_TIME_CHOICES[0])
PART_TIME = int(PART_TIME_CHOICES[0])
UNEMPLOYED = int(UNEMPLOYED_CHOICES[0])
RETIRED = int(RETIREMENT_CHOICES[0])
TABLE_ROW_LEN = 4  # label + 3 data columns


@pytask.mark.descriptives
@pytask.mark.summary_statistics
def task_describe_structural_estimation_sample(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "descriptives"
    / "structural_estimation_sample_summary.tex",
    restrict_to_all_observed: bool = False,
) -> None:
    """Create LaTeX table describing the structural estimation sample (women only).

    Three columns: Low education | High education | All.
    Panel A: unique households, unique individuals, person-year observations.
    Panel B: share full-time, part-time, unemployed, retired.
    Panel C: age (mean), share good health, share single (no partner).
    Panel D: average work experience, average wealth (in 1000 €).

    If restrict_to_all_observed is True, drop rows with any NaN across
    the variables used in the table. If False, allow NaNs where relevant.
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_full = pd.read_csv(path_to_data, index_col=[0])

    df = create_df_with_caregivers(
        df_full=df_full,
        specs=specs,
        start_year=2001,
        end_year=2019,
        end_age=specs["end_age_msm"],
    )

    # Women only (create_df_with_caregivers already filters sex==1; ensure)
    df = df.loc[df["sex"] == SEX].copy()

    cols_used = [
        "pid",
        "choice",
        "education",
        "health",
        "partner_state",
        "experience",
        "wealth",
    ]
    if restrict_to_all_observed:
        df = df.dropna(subset=[c for c in cols_used if c in df.columns])

    # Unique households: structural sample does not contain household id
    n_households = "n.a."

    def _stats(_df: pd.DataFrame) -> dict:
        if len(_df) == 0:
            return {
                "n_households": "n.a.",
                "n_individuals": 0,
                "n_person_years": 0,
                "share_ft": float("nan"),
                "share_pt": float("nan"),
                "share_unemp": float("nan"),
                "share_retired": float("nan"),
                "mean_age": float("nan"),
                "share_good_health": float("nan"),
                "share_single": float("nan"),
                "mean_experience": float("nan"),
                "mean_wealth_1000": float("nan"),
            }
        n_ind = _df["pid"].nunique()
        n_py = len(_df)
        share_ft = (_df["choice"] == FULL_TIME).mean()
        share_pt = (_df["choice"] == PART_TIME).mean()
        share_unemp = (_df["choice"] == UNEMPLOYED).mean()
        share_retired = (_df["choice"] == RETIRED).mean()
        mean_age = _df["age"].mean()
        share_good_health = (_df["health"] == GOOD_HEALTH).mean()
        # partner_state: 0 = no partner (single)
        share_single = (_df["partner_state"] == 0).mean()
        mean_exp = _df["experience"].mean()
        mean_wealth_1000 = _df["wealth"].mean() / 1_000.0
        return {
            "n_households": n_households,
            "n_individuals": n_ind,
            "n_person_years": n_py,
            "share_ft": share_ft,
            "share_pt": share_pt,
            "share_unemp": share_unemp,
            "share_retired": share_retired,
            "mean_age": mean_age,
            "share_good_health": share_good_health,
            "share_single": share_single,
            "mean_experience": mean_exp,
            "mean_wealth_1000": mean_wealth_1000,
        }

    low = df.loc[df["education"] == 0]
    high = df.loc[df["education"] == 1]
    s_low = _stats(low)
    s_high = _stats(high)
    s_all = _stats(df)

    def _fmt_num(x, decimals=2):
        if isinstance(x, (int, float)) and pd.isna(x):
            return "---"
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            if x == int(x):
                return str(int(x))
            return f"{x:.{decimals}f}"
        return str(x)

    def _fmt_pct(x):
        if isinstance(x, (int, float)) and pd.isna(x):
            return "---"
        return f"{100 * x:.1f}"

    rows = [
        ("Panel A", "", "", ""),
        (
            "\\quad Unique households",
            _fmt_num(s_low["n_households"]),
            _fmt_num(s_high["n_households"]),
            _fmt_num(s_all["n_households"]),
        ),
        (
            "\\quad Unique individuals",
            _fmt_num(s_low["n_individuals"]),
            _fmt_num(s_high["n_individuals"]),
            _fmt_num(s_all["n_individuals"]),
        ),
        (
            "\\quad Person-year observations",
            _fmt_num(s_low["n_person_years"]),
            _fmt_num(s_high["n_person_years"]),
            _fmt_num(s_all["n_person_years"]),
        ),
        ("Panel B", "", "", ""),
        (
            "\\quad Share full-time",
            _fmt_pct(s_low["share_ft"]),
            _fmt_pct(s_high["share_ft"]),
            _fmt_pct(s_all["share_ft"]),
        ),
        (
            "\\quad Share part-time",
            _fmt_pct(s_low["share_pt"]),
            _fmt_pct(s_high["share_pt"]),
            _fmt_pct(s_all["share_pt"]),
        ),
        (
            "\\quad Share unemployed",
            _fmt_pct(s_low["share_unemp"]),
            _fmt_pct(s_high["share_unemp"]),
            _fmt_pct(s_all["share_unemp"]),
        ),
        (
            "\\quad Share retired",
            _fmt_pct(s_low["share_retired"]),
            _fmt_pct(s_high["share_retired"]),
            _fmt_pct(s_all["share_retired"]),
        ),
        ("Panel C", "", "", ""),
        (
            "\\quad Age",
            _fmt_num(s_low["mean_age"]),
            _fmt_num(s_high["mean_age"]),
            _fmt_num(s_all["mean_age"]),
        ),
        (
            "\\quad Share good health",
            _fmt_pct(s_low["share_good_health"]),
            _fmt_pct(s_high["share_good_health"]),
            _fmt_pct(s_all["share_good_health"]),
        ),
        (
            "\\quad Share single (no partner)",
            _fmt_pct(s_low["share_single"]),
            _fmt_pct(s_high["share_single"]),
            _fmt_pct(s_all["share_single"]),
        ),
        ("Panel D", "", "", ""),
        (
            "\\quad Average work experience",
            _fmt_num(s_low["mean_experience"]),
            _fmt_num(s_high["mean_experience"]),
            _fmt_num(s_all["mean_experience"]),
        ),
        (
            "\\quad Average wealth (1000 €)",
            _fmt_num(s_low["mean_wealth_1000"]),
            _fmt_num(s_high["mean_wealth_1000"]),
            _fmt_num(s_all["mean_wealth_1000"]),
        ),
    ]

    header = (
        "\\begin{tabular}{llcc|c}\n"
        "\\toprule\n"
        " & & Low education & High education & All \\\\\n"
        "\\midrule\n"
    )
    body = [
        " & ".join(str(x) for x in row) + " \\\\"
        for row in rows
        if len(row) == TABLE_ROW_LEN
    ]
    body.append("\\bottomrule\n\\end{tabular}")

    latex = header + "\n".join(body) + "\n"
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    path_to_save_table.write_text(latex, encoding="utf-8")


def _care_ever_structural_sample(df: pd.DataFrame) -> pd.Series:
    """Create care_ever from any_care: True if person ever has any_care > 0.

    Same idea as in simulation (groupby agent, informal_care.any()) and
    task_plot_model_fit_estimated_params (groupby agent, is_care.agg(any)).
    For SOEP structural sample we have any_care (0/1) per person-year.
    """
    if "any_care" not in df.columns:
        return pd.Series(False, index=df.index)
    return df.groupby("pid")["any_care"].transform(lambda x: (x.fillna(0) > 0).any())


@pytask.mark.descriptives
@pytask.mark.summary_statistics
def task_describe_structural_estimation_sample_by_caregiver(  # noqa: PLR0912, PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "descriptives"
    / "structural_estimation_sample_summary_by_caregiver.tex",
    restrict_to_all_observed: bool = False,
) -> None:
    """Create LaTeX table: structural estimation sample by never/ever caregiver.

    Three columns: Never caregivers | Ever caregivers | All (women only).
    Same panels and variables as task_describe_structural_estimation_sample.
    care_ever is built from any_care (person ever has any_care > 0 in sample).
    """
    specs = pickle.load(path_to_specs.open("rb"))
    df_full = pd.read_csv(path_to_data, index_col=[0])

    df = create_df_with_caregivers(
        df_full=df_full,
        specs=specs,
        start_year=2001,
        end_year=2019,
        end_age=specs["end_age_msm"],
    )
    df = df.loc[df["sex"] == SEX].copy()

    cols_used = [
        "pid",
        "choice",
        "education",
        "health",
        "partner_state",
        "experience",
        "wealth",
        "any_care",
        "light_care",
        "intensive_care",
    ]
    if restrict_to_all_observed:
        df = df.dropna(subset=[c for c in cols_used if c in df.columns])

    df["care_ever"] = _care_ever_structural_sample(df)

    n_households = "n.a."

    def _stats(_df: pd.DataFrame) -> dict:
        if len(_df) == 0:
            return {
                "n_households": "n.a.",
                "n_individuals": 0,
                "n_person_years": 0,
                "share_ft": float("nan"),
                "share_pt": float("nan"),
                "share_unemp": float("nan"),
                "share_retired": float("nan"),
                "mean_age": float("nan"),
                "share_good_health": float("nan"),
                "share_single": float("nan"),
                "mean_experience": float("nan"),
                "mean_wealth_1000": float("nan"),
            }
        n_ind = _df["pid"].nunique()
        n_py = len(_df)
        share_ft = (_df["choice"] == FULL_TIME).mean()
        share_pt = (_df["choice"] == PART_TIME).mean()
        share_unemp = (_df["choice"] == UNEMPLOYED).mean()
        share_retired = (_df["choice"] == RETIRED).mean()
        mean_age = _df["age"].mean()
        share_good_health = (_df["health"] == GOOD_HEALTH).mean()
        share_single = (_df["partner_state"] == 0).mean()
        mean_exp = _df["experience"].mean()
        mean_wealth_1000 = _df["wealth"].mean() / 1_000.0
        return {
            "n_households": n_households,
            "n_individuals": n_ind,
            "n_person_years": n_py,
            "share_ft": share_ft,
            "share_pt": share_pt,
            "share_unemp": share_unemp,
            "share_retired": share_retired,
            "mean_age": mean_age,
            "share_good_health": share_good_health,
            "share_single": share_single,
            "mean_experience": mean_exp,
            "mean_wealth_1000": mean_wealth_1000,
        }

    never = df.loc[~df["care_ever"]]
    ever = df.loc[df["care_ever"]]
    s_never = _stats(never)
    s_ever = _stats(ever)
    s_all = _stats(df)
    # Person-years currently caregiving / not caregiving (for Panel B3 and Panel E)
    _ever_with_care = ever.loc[ever["any_care"] == 1]
    _all_with_care = df.loc[df["any_care"] == 1]
    _non_care = df.loc[
        df["any_care"] == 0
    ]  # current non-caregivers (for Panel B3 col 1)

    # Panel B2: labor shares by age bin; 60-69 includes retired (FT/PT/Unemp/Retired)
    _labor_choices = (PART_TIME, FULL_TIME, UNEMPLOYED)  # 2, 3, 1
    _age_bins = [(30, 40), (40, 50), (50, 60), (60, 70)]  # 30-39, 40-49, 50-59, 60-69

    def _labor_shares_by_age_bin(_df: pd.DataFrame, age_lo: int, age_hi: int) -> tuple:
        """Shares of FT, PT, Unemp conditional on labor force only (excl. retired)."""
        sub = _df[
            (_df["age"] >= age_lo)
            & (_df["age"] < age_hi)
            & (_df["choice"].isin(_labor_choices))
        ]
        if len(sub) == 0:
            return (float("nan"), float("nan"), float("nan"))
        return (
            (sub["choice"] == FULL_TIME).mean(),
            (sub["choice"] == PART_TIME).mean(),
            (sub["choice"] == UNEMPLOYED).mean(),
        )

    def _labor_shares_by_age_bin_incl_retired(
        _df: pd.DataFrame, age_lo: int, age_hi: int
    ) -> tuple:
        """Shares of FT, PT, Unemp, Retired among all in age bin."""
        sub = _df[(_df["age"] >= age_lo) & (_df["age"] < age_hi)]
        if len(sub) == 0:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        return (
            (sub["choice"] == FULL_TIME).mean(),
            (sub["choice"] == PART_TIME).mean(),
            (sub["choice"] == UNEMPLOYED).mean(),
            (sub["choice"] == RETIRED).mean(),
        )

    def _fmt_num(x, decimals=2):
        if isinstance(x, (int, float)) and pd.isna(x):
            return "---"
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            if x == int(x):
                return str(int(x))
            return f"{x:.{decimals}f}"
        return str(x)

    def _fmt_pct(x):
        if isinstance(x, (int, float)) and pd.isna(x):
            return "---"
        return f"{100 * x:.1f}"

    rows = [
        ("Panel A", "", "", ""),
        (
            "\\quad Unique households",
            _fmt_num(s_never["n_households"]),
            _fmt_num(s_ever["n_households"]),
            _fmt_num(s_all["n_households"]),
        ),
        (
            "\\quad Unique individuals",
            _fmt_num(s_never["n_individuals"]),
            _fmt_num(s_ever["n_individuals"]),
            _fmt_num(s_all["n_individuals"]),
        ),
        (
            "\\quad Person-year observations",
            _fmt_num(s_never["n_person_years"]),
            _fmt_num(s_ever["n_person_years"]),
            _fmt_num(s_all["n_person_years"]),
        ),
        ("Panel B", "", "", ""),
        (
            "\\quad Share full-time",
            _fmt_pct(s_never["share_ft"]),
            _fmt_pct(s_ever["share_ft"]),
            _fmt_pct(s_all["share_ft"]),
        ),
        (
            "\\quad Share part-time",
            _fmt_pct(s_never["share_pt"]),
            _fmt_pct(s_ever["share_pt"]),
            _fmt_pct(s_all["share_pt"]),
        ),
        (
            "\\quad Share unemployed",
            _fmt_pct(s_never["share_unemp"]),
            _fmt_pct(s_ever["share_unemp"]),
            _fmt_pct(s_all["share_unemp"]),
        ),
        (
            "\\quad Share retired",
            _fmt_pct(s_never["share_retired"]),
            _fmt_pct(s_ever["share_retired"]),
            _fmt_pct(s_all["share_retired"]),
        ),
        ("Panel B2 (labor only: FT, PT, unemp by age bin)", "", "", ""),
    ]
    for age_lo, age_hi in _age_bins:
        if (age_lo, age_hi) == (60, 70):
            lb_never = _labor_shares_by_age_bin_incl_retired(never, age_lo, age_hi)
            lb_ever = _labor_shares_by_age_bin_incl_retired(ever, age_lo, age_hi)
            lb_all = _labor_shares_by_age_bin_incl_retired(df, age_lo, age_hi)
            ft_n, pt_n, unemp_n, ret_n = lb_never
            ft_e, pt_e, unemp_e, ret_e = lb_ever
            ft_a, pt_a, unemp_a, ret_a = lb_all
        else:
            lb_never = _labor_shares_by_age_bin(never, age_lo, age_hi)
            lb_ever = _labor_shares_by_age_bin(ever, age_lo, age_hi)
            lb_all = _labor_shares_by_age_bin(df, age_lo, age_hi)
            ft_n, pt_n, unemp_n = lb_never
            ft_e, pt_e, unemp_e = lb_ever
            ft_a, pt_a, unemp_a = lb_all
            ret_n = ret_e = ret_a = None
        rows.append((f"\\quad Age {age_lo}--{age_hi - 1}", "", "", ""))
        rows.extend(
            (
                (
                    "\\quad \\quad Share full-time",
                    _fmt_pct(ft_n),
                    _fmt_pct(ft_e),
                    _fmt_pct(ft_a),
                ),
                (
                    "\\quad \\quad Share part-time",
                    _fmt_pct(pt_n),
                    _fmt_pct(pt_e),
                    _fmt_pct(pt_a),
                ),
                (
                    "\\quad \\quad Share unemployed",
                    _fmt_pct(unemp_n),
                    _fmt_pct(unemp_e),
                    _fmt_pct(unemp_a),
                ),
            )
        )
        if ret_n is not None:
            rows.append(
                (
                    "\\quad \\quad Share retired",
                    _fmt_pct(ret_n),
                    _fmt_pct(ret_e),
                    _fmt_pct(ret_a),
                )
            )

    # Panel B3: col 1 = current non-caregivers, col 2/3 = current caregivers
    rows.extend(
        (
            ("Panel B3 (when caregiving: FT / PT / Unemp by age bin)", "", "", ""),
            (
                "",
                "Current non-caregivers",
                "Current caregivers",
                "All",
            ),
        )
    )
    for age_lo, age_hi in _age_bins:
        if (age_lo, age_hi) == (60, 70):
            lb_non = _labor_shares_by_age_bin_incl_retired(_non_care, age_lo, age_hi)
            lb_ever_care = _labor_shares_by_age_bin_incl_retired(
                _ever_with_care, age_lo, age_hi
            )
            lb_all_care = _labor_shares_by_age_bin_incl_retired(
                _all_with_care, age_lo, age_hi
            )
            ft_n, pt_n, unemp_n, ret_n = lb_non
            ft_e, pt_e, unemp_e, ret_e = lb_ever_care
            ft_a, pt_a, unemp_a, ret_a = lb_all_care
        else:
            lb_non = _labor_shares_by_age_bin(_non_care, age_lo, age_hi)
            lb_ever_care = _labor_shares_by_age_bin(_ever_with_care, age_lo, age_hi)
            lb_all_care = _labor_shares_by_age_bin(_all_with_care, age_lo, age_hi)
            ft_n, pt_n, unemp_n = lb_non
            ft_e, pt_e, unemp_e = lb_ever_care
            ft_a, pt_a, unemp_a = lb_all_care
            ret_n = ret_e = ret_a = None
        rows.append((f"\\quad Age {age_lo}--{age_hi - 1}", "", "", ""))
        rows.extend(
            (
                (
                    "\\quad \\quad Share full-time",
                    _fmt_pct(ft_n),
                    _fmt_pct(ft_e),
                    _fmt_pct(ft_a),
                ),
                (
                    "\\quad \\quad Share part-time",
                    _fmt_pct(pt_n),
                    _fmt_pct(pt_e),
                    _fmt_pct(pt_a),
                ),
                (
                    "\\quad \\quad Share unemployed",
                    _fmt_pct(unemp_n),
                    _fmt_pct(unemp_e),
                    _fmt_pct(unemp_a),
                ),
            )
        )
        if ret_n is not None:
            rows.append(
                (
                    "\\quad \\quad Share retired",
                    _fmt_pct(ret_n),
                    _fmt_pct(ret_e),
                    _fmt_pct(ret_a),
                )
            )

    rows.extend(
        [
            ("Panel C", "", "", ""),
            (
                "\\quad Age",
                _fmt_num(s_never["mean_age"]),
                _fmt_num(s_ever["mean_age"]),
                _fmt_num(s_all["mean_age"]),
            ),
            (
                "\\quad Share good health",
                _fmt_pct(s_never["share_good_health"]),
                _fmt_pct(s_ever["share_good_health"]),
                _fmt_pct(s_all["share_good_health"]),
            ),
            (
                "\\quad Share single (no partner)",
                _fmt_pct(s_never["share_single"]),
                _fmt_pct(s_ever["share_single"]),
                _fmt_pct(s_all["share_single"]),
            ),
            ("Panel D", "", "", ""),
            (
                "\\quad Average work experience",
                _fmt_num(s_never["mean_experience"]),
                _fmt_num(s_ever["mean_experience"]),
                _fmt_num(s_all["mean_experience"]),
            ),
            (
                "\\quad Average wealth (1000 €)",
                _fmt_num(s_never["mean_wealth_1000"]),
                _fmt_num(s_ever["mean_wealth_1000"]),
                _fmt_num(s_all["mean_wealth_1000"]),
            ),
        ]
    )

    # Panel E: caregivers only (ever caregivers)
    if len(ever) > 0:
        _age_first_care = ever.loc[ever["any_care"] == 1].groupby("pid")["age"].min()
        _care_years = ever.groupby("pid")["any_care"].sum()
        _mean_age_first = _age_first_care.mean()
        _std_age_first = _age_first_care.std()
        _mean_care_years = _care_years.mean()
        _std_care_years = _care_years.std()
        if len(_ever_with_care) > 0:
            _share_intensive = (_ever_with_care["intensive_care"] == 1).mean()
            _share_light = (_ever_with_care["light_care"] == 1).mean()
            _mean_age_when_caregiving = _ever_with_care["age"].mean()
            _std_age_when_caregiving = _ever_with_care["age"].std()
        else:
            _share_intensive = float("nan")
            _share_light = float("nan")
            _mean_age_when_caregiving = float("nan")
            _std_age_when_caregiving = float("nan")
    else:
        _mean_age_first = _std_age_first = float("nan")
        _mean_care_years = _std_care_years = float("nan")
        _share_intensive = _share_light = float("nan")
        _mean_age_when_caregiving = _std_age_when_caregiving = float("nan")
    if len(_all_with_care) > 0:
        _mean_age_when_caregiving_all = _all_with_care["age"].mean()
        _std_age_when_caregiving_all = _all_with_care["age"].std()
    else:
        _mean_age_when_caregiving_all = _std_age_when_caregiving_all = float("nan")

    def _fmt_mean_std(mean_val, std_val, decimals=1):
        if pd.isna(mean_val) or pd.isna(std_val):
            return "---"
        return f"{mean_val:.{decimals}f} ({std_val:.{decimals}f})"

    rows.extend(
        [
            ("Panel E (caregivers only)", "", "", ""),
            (
                "\\quad Average age at first care spell",
                "---",
                _fmt_mean_std(_mean_age_first, _std_age_first),
                _fmt_mean_std(_mean_age_first, _std_age_first),
            ),
            (
                "\\quad Average age (when caregiving)",
                "---",
                _fmt_mean_std(_mean_age_when_caregiving, _std_age_when_caregiving),
                _fmt_mean_std(
                    _mean_age_when_caregiving_all, _std_age_when_caregiving_all
                ),
            ),
            (
                "\\quad Average number of care years",
                "---",
                _fmt_mean_std(_mean_care_years, _std_care_years),
                _fmt_mean_std(_mean_care_years, _std_care_years),
            ),
            (
                "\\quad Share intensive caregiving",
                "---",
                _fmt_pct(_share_intensive),
                _fmt_pct(_share_intensive),
            ),
            (
                "\\quad Share light caregiving",
                "---",
                _fmt_pct(_share_light),
                _fmt_pct(_share_light),
            ),
        ]
    )

    header = (
        "\\begin{tabular}{llcc|c}\n"
        "\\toprule\n"
        " & & Never caregivers & Ever caregivers & All \\\\\n"
        "\\midrule\n"
    )
    body = [
        " & ".join(str(x) for x in row) + " \\\\"
        for row in rows
        if len(row) == TABLE_ROW_LEN
    ]
    body.append("\\bottomrule\n\\end{tabular}")

    latex = header + "\n".join(body) + "\n"
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    path_to_save_table.write_text(latex, encoding="utf-8")


@pytask.mark.descriptives
@pytask.mark.summary_statistics
def task_describe_structural_estimation_sample_by_caregiver_v2(  # noqa: PLR0912, PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_save_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "descriptives"
    / "structural_estimation_sample_summary_by_caregiver_v2.tex",
    restrict_to_all_observed: bool = False,
) -> None:
    """Restructured descriptive table: never/ever/all block, then non-CG/CG block."""
    specs = pickle.load(path_to_specs.open("rb"))
    df_full = pd.read_csv(path_to_data, index_col=[0])

    df = create_df_with_caregivers(
        df_full=df_full,
        specs=specs,
        start_year=2001,
        end_year=2019,
        end_age=specs["end_age_msm"],
    )
    df = df.loc[df["sex"] == SEX].copy()

    cols_used = [
        "pid",
        "choice",
        "education",
        "health",
        "partner_state",
        "experience",
        "wealth",
        "any_care",
        "light_care",
        "intensive_care",
    ]
    if restrict_to_all_observed:
        df = df.dropna(subset=[c for c in cols_used if c in df.columns])

    df["care_ever"] = _care_ever_structural_sample(df)

    def _stats_v2(_df: pd.DataFrame) -> dict:
        if len(_df) == 0:
            return {
                "n_individuals": 0,
                "n_person_years": 0,
                "share_ft": float("nan"),
                "share_pt": float("nan"),
                "share_unemp": float("nan"),
                "share_retired": float("nan"),
                "mean_age": float("nan"),
                "share_good_health": float("nan"),
                "share_single": float("nan"),
                "mean_experience": float("nan"),
                "mean_wealth_1000": float("nan"),
            }
        return {
            "n_individuals": _df["pid"].nunique(),
            "n_person_years": len(_df),
            "share_ft": (_df["choice"] == FULL_TIME).mean(),
            "share_pt": (_df["choice"] == PART_TIME).mean(),
            "share_unemp": (_df["choice"] == UNEMPLOYED).mean(),
            "share_retired": (_df["choice"] == RETIRED).mean(),
            "mean_age": _df["age"].mean(),
            "share_good_health": (_df["health"] == GOOD_HEALTH).mean(),
            "share_single": (_df["partner_state"] == 0).mean(),
            "mean_experience": _df["experience"].mean(),
            "mean_wealth_1000": _df["wealth"].mean() / 1_000.0,
        }

    _labor_choices_v2 = (PART_TIME, FULL_TIME, UNEMPLOYED)
    _age_bins_v2 = [(30, 40), (40, 50), (50, 60), (60, 70)]

    def _labor_by_bin(_df: pd.DataFrame, age_lo: int, age_hi: int) -> tuple:
        sub = _df[(_df["age"] >= age_lo) & (_df["age"] < age_hi)]
        if (age_lo, age_hi) == (60, 70):
            if len(sub) == 0:
                return (float("nan"),) * 4
            return (
                (sub["choice"] == FULL_TIME).mean(),
                (sub["choice"] == PART_TIME).mean(),
                (sub["choice"] == UNEMPLOYED).mean(),
                (sub["choice"] == RETIRED).mean(),
            )
        sub = sub[sub["choice"].isin(_labor_choices_v2)]
        if len(sub) == 0:
            return (float("nan"),) * 3
        return (
            (sub["choice"] == FULL_TIME).mean(),
            (sub["choice"] == PART_TIME).mean(),
            (sub["choice"] == UNEMPLOYED).mean(),
        )

    def _fmt_n(x, decimals=2):
        if isinstance(x, (int, float)) and pd.isna(x):
            return "---"
        if isinstance(x, int):
            return f"{x:,}"
        if isinstance(x, float):
            if x == int(x):
                return f"{int(x):,}"
            return f"{x:.{decimals}f}"
        return str(x)

    def _fmt_p(x):
        if isinstance(x, (int, float)) and pd.isna(x):
            return "---"
        return f"{100 * x:.1f}"

    def _fmt_ms(mean_val, std_val, decimals=1):
        if pd.isna(mean_val) or pd.isna(std_val):
            return "---"
        return f"{mean_val:.{decimals}f} ({std_val:.{decimals}f})"

    never = df.loc[~df["care_ever"]]
    ever = df.loc[df["care_ever"]]
    s_never = _stats_v2(never)
    s_ever = _stats_v2(ever)
    s_all = _stats_v2(df)

    # ---- Block 1: Never | Ever | All (3 columns) ----
    b1 = []

    b1.extend(
        (
            r"\multicolumn{4}{l}{\textit{Panel A: Sample}} \\",
            f"\\quad Unique individuals & {_fmt_n(s_never['n_individuals'])} "
            f"& {_fmt_n(s_ever['n_individuals'])} "
            f"& {_fmt_n(s_all['n_individuals'])} \\\\",
        )
    )
    b1.append(
        f"\\quad Person-year observations & {_fmt_n(s_never['n_person_years'])} "
        f"& {_fmt_n(s_ever['n_person_years'])} & {_fmt_n(s_all['n_person_years'])} \\\\"
    )

    b1.extend(
        (
            r"\addlinespace",
            r"\multicolumn{4}{l}{\textit{Panel B: Labor market status"
            r" (all ages, \%)}} \\",
        )
    )
    for label, key in (
        ("Share full-time", "share_ft"),
        ("Share part-time", "share_pt"),
        ("Share non-employed", "share_unemp"),
        ("Share retired", "share_retired"),
    ):
        b1.append(
            f"\\quad {label} & {_fmt_p(s_never[key])} "
            f"& {_fmt_p(s_ever[key])} & {_fmt_p(s_all[key])} \\\\"
        )

    b1.extend(
        (
            r"\addlinespace",
            r"\multicolumn{4}{l}{\textit{Panel B2: Labor market status"
            r" by age bin (\%)}} \\",
        )
    )
    for age_lo, age_hi in _age_bins_v2:
        lb_n = _labor_by_bin(never, age_lo, age_hi)
        lb_e = _labor_by_bin(ever, age_lo, age_hi)
        lb_a = _labor_by_bin(df, age_lo, age_hi)
        b1.append(f"\\quad Age {age_lo}--{age_hi - 1} & & & \\\\")
        incl_ret = (age_lo, age_hi) == (60, 70)
        labels = ["Share full-time", "Share part-time", "Share non-employed"]
        if incl_ret:
            labels.append("Share retired")
        for i, lab in enumerate(labels):
            b1.append(
                f"\\quad \\quad {lab} & {_fmt_p(lb_n[i])} "
                f"& {_fmt_p(lb_e[i])} & {_fmt_p(lb_a[i])} \\\\"
            )

    b1.extend(
        (r"\addlinespace", r"\multicolumn{4}{l}{\textit{Panel C: Demographics}} \\")
    )
    b1.extend(
        (
            f"\\quad Average age & {_fmt_n(s_never['mean_age'])} "
            f"& {_fmt_n(s_ever['mean_age'])} & {_fmt_n(s_all['mean_age'])} \\\\",
            f"\\quad Share good health & {_fmt_p(s_never['share_good_health'])} "
            f"& {_fmt_p(s_ever['share_good_health'])} "
            f"& {_fmt_p(s_all['share_good_health'])} \\\\",
        )
    )
    b1.append(
        f"\\quad Share single (no partner) & {_fmt_p(s_never['share_single'])} "
        f"& {_fmt_p(s_ever['share_single'])} "
        f"& {_fmt_p(s_all['share_single'])} \\\\"
    )

    b1.extend(
        (
            r"\addlinespace",
            r"\multicolumn{4}{l}{\textit{Panel D: Economic outcomes}} \\",
        )
    )
    b1.extend(
        (
            f"\\quad Average work experience & {_fmt_n(s_never['mean_experience'])} "
            f"& {_fmt_n(s_ever['mean_experience'])} "
            f"& {_fmt_n(s_all['mean_experience'])} \\\\",
            f"\\quad Average wealth (1000 EUR) & {_fmt_n(s_never['mean_wealth_1000'])} "
            f"& {_fmt_n(s_ever['mean_wealth_1000'])} "
            f"& {_fmt_n(s_all['mean_wealth_1000'])} \\\\",
        )
    )

    # ---- Block 2: Current non-CG | Current CG (2 columns) ----
    _non_care = df.loc[df["any_care"] == 0]
    _cur_care = df.loc[df["any_care"] == 1]

    b2 = []
    b2.extend((r"\addlinespace", r"\midrule"))
    b2.extend((" & Current non-caregivers & Current caregivers & \\\\", r"\midrule"))
    b2.append(
        r"\multicolumn{4}{l}{\textit{Panel B3: Labor market status during"
        r" caregiving by age bin (\%)}} \\"
    )
    for age_lo, age_hi in _age_bins_v2:
        lb_nc = _labor_by_bin(_non_care, age_lo, age_hi)
        lb_cc = _labor_by_bin(_cur_care, age_lo, age_hi)
        b2.append(f"\\quad Age {age_lo}--{age_hi - 1} & & & \\\\")
        incl_ret = (age_lo, age_hi) == (60, 70)
        labels = ["Share full-time", "Share part-time", "Share non-employed"]
        if incl_ret:
            labels.append("Share retired")
        for i, lab in enumerate(labels):
            b2.append(
                f"\\quad \\quad {lab} & {_fmt_p(lb_nc[i])} & {_fmt_p(lb_cc[i])} & \\\\"
            )

    # ---- Panel E: Caregivers only ----
    b3 = []
    b3.extend(
        (r"\addlinespace", r"\multicolumn{4}{l}{\textit{Panel E: Caregivers only}} \\")
    )

    if len(ever) > 0:
        _ever_with_care = ever.loc[ever["any_care"] == 1]
        if len(_ever_with_care) > 0:
            _age_first = _ever_with_care.groupby("pid")["age"].min()
        else:
            _age_first = pd.Series(dtype=float)
        _care_yrs = ever.groupby("pid")["any_care"].sum()
        mean_af = _age_first.mean() if len(_age_first) > 0 else float("nan")
        std_af = _age_first.std() if len(_age_first) > 0 else float("nan")
        mean_cy = _care_yrs.mean()
        std_cy = _care_yrs.std()
        if len(_ever_with_care) > 0:
            sh_int = (_ever_with_care["intensive_care"] == 1).mean()
            sh_lt = (_ever_with_care["light_care"] == 1).mean()
            mean_acg = _ever_with_care["age"].mean()
            std_acg = _ever_with_care["age"].std()
        else:
            sh_int = sh_lt = mean_acg = std_acg = float("nan")
    else:
        mean_af = std_af = mean_cy = std_cy = float("nan")
        sh_int = sh_lt = mean_acg = std_acg = float("nan")

    b3.extend(
        (
            f"\\quad Average age at first care spell & --- "
            f"& {_fmt_ms(mean_af, std_af)} & \\\\",
            f"\\quad Average age (when caregiving) & --- "
            f"& {_fmt_ms(mean_acg, std_acg)} & \\\\",
        )
    )
    b3.extend(
        (
            f"\\quad Average number of care years & --- "
            f"& {_fmt_ms(mean_cy, std_cy)} & \\\\",
            f"\\quad Share intensive caregiving & --- & {_fmt_p(sh_int)} & \\\\",
        )
    )
    b3.append(f"\\quad Share light caregiving & --- & {_fmt_p(sh_lt)} & \\\\")

    # ---- Assemble ----
    header = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        " & Never caregivers & Ever caregivers & All \\\\\n"
        "\\midrule\n"
    )
    footer = "\\bottomrule\n\\end{tabular}\n"

    body = "\n".join(b1 + b2 + b3)
    latex = header + body + "\n" + footer
    path_to_save_table.parent.mkdir(parents=True, exist_ok=True)
    path_to_save_table.write_text(latex, encoding="utf-8")
