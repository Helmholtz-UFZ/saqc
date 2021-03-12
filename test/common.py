#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import numpy as np
import pandas as pd
import dios

from saqc.common import *
from saqc.flagger import Flagger, initFlagsLike


TESTNODATA = (np.nan, -9999)
TESTFLAGGER = (Flagger(),)


def flagAll(data, field, flagger, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    flagger.copy()
    flagger[:, field] = BAD
    return data, flagger


def initData(cols=2, start_date="2017-01-01", end_date="2017-12-31", freq=None, rows=None):
    if rows is None:
        freq = freq or "1h"

    di = dios.DictOfSeries(itype=dios.DtItype)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq, periods=rows)
    dummy = np.arange(len(dates))

    for col in range(1, cols + 1):
        di[f"var{col}"] = pd.Series(data=dummy * col, index=dates)

    return di


def writeIO(content):
    f = io.StringIO()
    f.write(content)
    f.seek(0)
    return f


