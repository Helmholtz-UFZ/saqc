#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod

__all__ = [
    "T",
    "ArrayLike",
    "PandasLike",
    "DiosLikeT",
    "CurveFitter",
    "ExternalFlag",
    "OptionalNone",
]


from typing import Any, Dict, TypeVar, Union

import numpy as np
import pandas as pd
from typing_extensions import Protocol

from dios import DictOfSeries

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)
PandasLike = Union[pd.Series, pd.DataFrame, DictOfSeries]
DiosLikeT = Union[DictOfSeries, pd.DataFrame]

ExternalFlag = Union[str, float, int]


# needed for deeper type hinting magic
class CurveFitter(Protocol):
    def __call__(self, data: np.ndarray, *params: float) -> np.ndarray:
        ...  # pragma: no cover


class GenericFunction(Protocol):

    __name__: str
    __globals__: Dict[str, Any]

    def __call__(self, *args: pd.Series) -> PandasLike:
        ...  # pragma: no cover


class Comparable(Protocol):
    @abstractmethod
    def __gt__(self: CompT, other: CompT) -> bool:
        pass


CompT = TypeVar("CompT", bound=Comparable)


class OptionalNone:
    pass
