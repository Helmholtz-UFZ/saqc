#! /usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = [
    "T",
    "ArrayLike",
    "PandasLike",
    "DiosLikeT",
    "FreqString",
    "IntegerWindow",
    "Timestampstr",
    "CurveFitter",
    "ExternalFlag",
    "PositiveFloat",
    "PositiveInt",
]

from typing import TypeVar, Union, NewType, List, Tuple
from typing_extensions import Protocol, Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)
PandasLike = Union[pd.Series, pd.DataFrame, DictOfSeries]
DiosLikeT = Union[DictOfSeries, pd.DataFrame]

ExternalFlag = Union[str, float, int]

# we only support fixed length offsets
FreqString = Literal["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]

# # we define a bunch of type aliases, mostly needed to generate appropiate fuzzy data through hypothesis
# ColumnName = NewType("ColumnName", str)
# IntegerWindow = NewType("IntegerWindow", int)
# Timestampstr = TypeVar("Timestampstr", bound=str)
# PositiveFloat = NewType("PositiveFloat", float)
# PositiveInt = NewType("PositiveInt", int)

# needed for deeper type hinting magic
class CurveFitter(Protocol):
    def __call__(self, data: np.ndarray, *params: float) -> np.ndarray:
        ...
