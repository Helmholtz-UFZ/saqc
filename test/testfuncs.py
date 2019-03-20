#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def initData(start_date="2017-01-01", end_date="2017-12-31", freq="1h"):
    dates = pd.date_range(start="2017-01-01", end="2017-12-31", freq="1h")
    data = pd.DataFrame(
        data={"var1": np.arange(len(dates)),
              "var2": np.arange(len(dates), len(dates)*2)},
        index=dates)
    return data
