#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import numpy as np
import pandas as pd
import dios

from saqc.constants import *
from saqc.core import initFlagsLike, Flags as Flagger


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


def checkDataFlaggerInvariants(data, flagger, field, identical=True):
    """
    Check all invariants that must hold at any point for
        * field
        * data
        * flagger
        * data[field]
        * flagger[field]
        * data[field].index
        * flagger[field].index
        * between data and flagger
        * between data[field] and flagger[field]

    Parameters
    ----------
    data : dios.DictOfSeries
        data container
    flagger : Flags
        flags container
    field : str
        the field in question
    identical : bool, default True
        whether to check indexes of data and flagger to be
        identical (True, default) of just for equality.
    """
    assert isinstance(data, dios.DictOfSeries)
    assert isinstance(flagger, Flagger)

    # all columns in data are in flagger
    assert data.columns.difference(flagger.columns).empty

    # ------------------------------------------------------------------------
    # below here, we just check on and with field
    # ------------------------------------------------------------------------
    assert field in data
    assert field in flagger

    assert flagger[field].dtype == float

    # `pd.Index.identical` also check index attributes like `freq`
    if identical:
        assert data[field].index.identical(flagger[field].index)
    else:
        assert data[field].index.equals(flagger[field].index)


