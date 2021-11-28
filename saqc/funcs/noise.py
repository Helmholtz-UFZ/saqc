#! /usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import operator
from dios import DictOfSeries
from typing import Callable
from saqc.constants import *
from saqc.core import register, Flags
from saqc.core.register import flagging
from saqc.lib.tools import statPass


@flagging()
def flagByStatLowPass(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    func: Callable[[np.array, pd.Series], float],
    window: str,
    thresh: float,
    sub_window: str = None,
    sub_thresh: float = None,
    min_periods: int = None,
    flag: float = BAD,
    **kwargs,
):
    """
    Flag *chunks* of length, `window`:

    1. If they excexceed `thresh` with regard to `stat`:
    2. If all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_window`,
       `excexceed `sub_thresh` with regard to `stat`:

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store quality flags to data.
    func: Callable[[np.array, pd.Series], float]
        Function to aggregate chunk contnent with.
    window: str
        Temporal extension of the chunks to test
    thresh: float
        Threshold, that triggers flagging, if exceeded by stat value.
    sub_window: str, default None,
        Window size of the sub chunks, that are additionally tested for exceeding
        `sub_thresh` with respect to `stat`.
    sub_thresh: float, default None
    min_periods: int, default None
    flag : float, default BAD
        flag to set

    Returns
    -------
    """

    datcol = data[field]
    if not min_periods:
        min_periods = 0
    if not sub_thresh:
        sub_thresh = thresh
    window = pd.Timedelta(window)

    if sub_window:
        sub_window = pd.Timedelta(sub_window)

    to_set = statPass(
        datcol, func, window, thresh, operator.gt, sub_window, sub_thresh, min_periods
    )
    flags[to_set, field] = flag
    return data, flags
