#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from typing import Any, Dict, TypeVar, Union

import numpy as np
import pandas as pd
from typing_extensions import Protocol

__all__ = [
    "T",
    "ArrayLike",
    "CurveFitter",
    "ExternalFlag",
    "OptionalNone",
]

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)

ExternalFlag = Union[str, float, int]


# needed for deeper type hinting magic
class CurveFitter(Protocol):
    def __call__(self, data: np.ndarray, *params: float) -> np.ndarray:
        ...  # pragma: no cover


class Comparable(Protocol):
    @abc.abstractmethod
    def __gt__(self: CompT, other: CompT) -> bool:
        pass


CompT = TypeVar("CompT", bound=Comparable)


class OptionalNone:
    pass
