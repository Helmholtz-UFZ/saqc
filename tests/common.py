#! /usr/bin/env python
# -*- coding: utf-8 -*-

import io
import numpy as np
import pandas as pd
import dios

from saqc.constants import *
from saqc.core import Flags
from saqc.core.history import History, createHistoryFromData


def flagAll(data, field, flags, **kwargs):
    # NOTE: remember to rename flag -> flag_values
    flags.copy()
    flags[:, field] = BAD
    return data, flags


def initData(
    cols=2, start_date="2017-01-01", end_date="2017-12-31", freq=None, rows=None
):
    if rows is None:
        freq = freq or "1h"

    di = dios.DictOfSeries(itype=dios.DtItype)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq, periods=rows)
    dummy = np.arange(len(dates))

    for col in range(1, cols + 1):
        di[f"var{col}"] = pd.Series(data=dummy * col, index=dates)

    return di


def dummyHistory(hist: pd.DataFrame = None, meta: list = None):
    if hist is None:
        return History()

    if meta is None:
        meta = [{}] * len(hist.columns)

    return createHistoryFromData(hist, meta, copy=True)


def writeIO(content):
    f = io.StringIO()
    f.write(content)
    f.seek(0)
    return f


def checkDataFlagsInvariants(data, flags, field, identical=True):
    """
    Check all invariants that must hold at any point for
        * field
        * data
        * flags
        * data[field]
        * flags[field]
        * data[field].index
        * flags[field].index
        * between data and flags
        * between data[field] and flags[field]

    Parameters
    ----------
    data : dios.DictOfSeries
        data container
    flags : Flags
        flags container
    field : str
        the field in question
    identical : bool, default True
        whether to check indexes of data and flags to be
        identical (True, default) of just for equality.
    """
    assert isinstance(data, dios.DictOfSeries)
    assert isinstance(flags, Flags)

    # all columns in data are in flags
    assert data.columns.difference(flags.columns).empty

    # ------------------------------------------------------------------------
    # below here, we just check on and with field
    # ------------------------------------------------------------------------
    assert field in data
    assert field in flags

    assert flags[field].dtype == float

    # `pd.Index.identical` also check index attributes like `freq`
    if identical:
        assert data[field].index.identical(flags[field].index)
    else:
        assert data[field].index.equals(flags[field].index)
