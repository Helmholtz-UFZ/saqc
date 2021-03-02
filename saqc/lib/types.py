#! /usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = [
    'T',
    'ArrayLike',
    'PandasLike',
    'DiosLikeT',
    'FuncReturnT',
    'FreqString',
    'ColumnName',
    'IntegerWindow',
    'TimestampColumnName',
    'CurveFitter',
]

from typing import TypeVar, Union, NewType
from typing_extensions import Protocol, Literal

import numpy as np
import pandas as pd
from dios import DictOfSeries
from saqc import Flagger

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)
PandasLike = TypeVar("PandasLike", pd.Series, pd.DataFrame, DictOfSeries)
DiosLikeT = Union[DictOfSeries, pd.DataFrame]

FuncReturnT = [DictOfSeries, Flagger]

# we only support fixed length offsets
FreqString = NewType("FreqString", Literal["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"])

# we define a bunch of type aliases, mostly needed to generate appropiate fuzzy data through hypothesis
ColumnName = NewType("ColumnName", str)
IntegerWindow = NewType("IntegerWindow", int)
TimestampColumnName = TypeVar("TimestampColumnName", bound=str)

# needed for deeper typy hinting magic
class CurveFitter(Protocol):
    def __call__(self, data: np.ndarray, *params: float) -> np.ndarray:
        ...

