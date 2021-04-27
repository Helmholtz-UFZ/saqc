#! /usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import operator
from dios import DictOfSeries
from typing import Callable
from saqc.constants import *
from saqc.core import register, Flags
from saqc.lib.types import ColumnName, FreqString, PositiveInt, PositiveFloat, Literal
from saqc.lib.tools import statPass


@register(masking="field", module="noise")
def flagByStatLowPass(
    data: DictOfSeries,
    field: ColumnName,
    flags: Flags,
    stat: Callable[[np.array, pd.Series], float],
    winsz: FreqString,
    thresh: PositiveFloat,
    sub_winsz: FreqString = None,
    sub_thresh: PositiveFloat = None,
    min_periods: PositiveInt = None,
    flag: float = BAD,
    **kwargs
):
    """
    Flag *chunks* of length, `winsz`:

    1. If they excexceed `thresh` with regard to `stat`:
    2. If all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_winsz`,
       `excexceed `sub_thresh` with regard to `stat`:

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store quality flags to data.
    stat: Callable[[np.array, pd.Series], float]
        Function to aggregate chunk contnent with.
    winsz: FreqString
        Temporal extension of the chunks to test
    thresh: PositiveFloat
        Threshold, that triggers flagging, if exceeded by stat value.
    sub_winsz: FreqString, default None,
        Window size of the sub chunks, that are additionally tested for exceeding
        `sub_thresh` with respect to `stat`.
    sub_thresh: PositiveFloat, default None
    min_periods: PositiveInt, default None

    Returns
    -------
    """

    datcol = data[field]
    if not min_periods:
        min_periods = 0
    if not sub_thresh:
        sub_thresh = thresh
    winsz = pd.Timedelta(winsz)

    if sub_winsz:
        sub_winsz = pd.Timedelta(sub_winsz)

    to_set = statPass(
        datcol, stat, winsz, thresh, operator.gt, sub_winsz, sub_thresh, min_periods
    )
    flags[to_set[to_set].index, field] = flag
    return data, flags
