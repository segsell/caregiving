"""This module contains the general configuration of the project."""

from pathlib import Path

JET_COLOR_MAP = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
FIGSIZE = 12

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()
BLD = ROOT.joinpath("bld").resolve()
PUBLIC = BLD.joinpath("public").resolve()
TESTS = ROOT.joinpath("tests").resolve()


__all__ = ["BLD", "SRC", "TESTS"]
