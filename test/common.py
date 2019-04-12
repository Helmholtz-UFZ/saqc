#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def initData(cols=2, start_date="2017-01-01", end_date="2017-12-31", freq="1h"):
    dates = pd.date_range(start="2017-01-01", end="2017-12-31", freq="1h")
    data = {}
    dummy = np.arange(len(dates))
    for col in range(1, cols+1):
        data[f"var{col}"] = dummy*(col)
    return pd.DataFrame(data, index=dates)
