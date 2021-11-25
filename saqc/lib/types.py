#! /usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = [
    "T",
    "ArrayLike",
    "PandasLike",
    "DiosLikeT",
    "CurveFitter",
    "ExternalFlag",
    "OptionalNone",
]

from typing import Any, TypeVar, Union, Dict
from typing_extensions import Protocol, Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)
PandasLike = Union[pd.Series, pd.DataFrame, DictOfSeries]
DiosLikeT = Union[DictOfSeries, pd.DataFrame]

ExternalFlag = Union[str, float, int]


# needed for deeper type hinting magic
class CurveFitter(Protocol):
    def __call__(self, data: np.ndarray, *params: float) -> np.ndarray:
        ...


class GenericFunction(Protocol):

    __name__: str
    __globals__: Dict[str, Any]

    def __call__(self, *args: pd.Series) -> PandasLike:
        ...


class OptionalNone:
    pass
