#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TypeVar, Union

import numpy as np
import pandas as pd
import dios
from saqc.flagger.flags import Flagger

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)
PandasLike = TypeVar("PandasLike", pd.Series, pd.DataFrame, dios.DictOfSeries)
DiosLikeT = Union[dios.DictOfSeries, pd.DataFrame]

FuncReturnT = [dios.DictOfSeries, Flagger]
