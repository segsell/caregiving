"""Create a LaTeX table of average annual and weekly working hours by education."""

import pickle
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD

WEEKS_PER_YEAR = 52
SEX_WOMEN = 1


@pytask.mark.pre_estimation
def task_working_hours_table(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "tables"
    / "pre_estimation"
    / "working_hours_by_education.tex",
) -> None:
    specs = pickle.load(path_to_specs.open("rb"))
    edu_labels = specs["education_labels"]
    pt = specs["av_annual_hours_pt"][SEX_WOMEN]
    ft = specs["av_annual_hours_ft"][SEX_WOMEN]

    rows = []
    for i, label in enumerate(edu_labels):
        rows.append(
            {
                "Education": label,
                "Part-time (annual)": float(pt[i]),
                "Full-time (annual)": float(ft[i]),
                "Part-time (weekly)": float(pt[i]) / WEEKS_PER_YEAR,
                "Full-time (weekly)": float(ft[i]) / WEEKS_PER_YEAR,
            }
        )

    df = pd.DataFrame(rows).set_index("Education")
    latex_str = df.to_latex(float_format="%.1f")
    path_to_save.write_text(latex_str)
