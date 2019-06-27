#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")
ArrayLike = TypeVar("ArrayLike", np.ndarray, pd.Series, pd.DataFrame)
PandasLike = TypeVar("PandasLike", pd.Series, pd.DataFrame)
