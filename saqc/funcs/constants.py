#! /usr/bin/env python
# -*- coding: utf-8 -*-

from operator import mod
from typing import Tuple

import numpy as np
import pandas as pd
import operator

from dios import DictOfSeries

from saqc.constants import *
from saqc.core import flagging, Flags
from saqc.lib.ts_operators import varQC
from saqc.lib.tools import customRoller, getFreqDelta, statPass
from saqc.lib.types import FreqString


@flagging(masking="field")
def flagConstants(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    thresh: float,
    window: FreqString,
    flag: float = BAD,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    This functions flags plateaus/series of constant values of length `window` if
    their maximum total change is smaller than thresh.

    Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:

    (1) n > `window`
    (2) |(y(t + i) - (t + j)| < `thresh`, for all i,j in [0, 1, ..., n]

    Flag values are (semi-)constant.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        Name of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store quality flags to data.
    thresh : float
        Upper bound for the maximum total change of an interval to be flagged constant.
    window : str
        Lower bound for the size of an interval to be flagged constant.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The flags object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flags input.
    """
    if not isinstance(window, str):
        raise TypeError("window must be offset string.")
    d = data[field]

    # min_periods=2 ensures that at least two non-nan values are present
    # in each window and also min() == max() == d[i] is not possible.
    kws = dict(window=window, min_periods=2, expand=False)

    # 1. find starting points of consecutive constant values as a boolean mask
    # 2. fill the whole window with True's
    rolling = customRoller(d, **kws)
    starting_points_mask = rolling.max() - rolling.min() <= thresh
    rolling = customRoller(starting_points_mask, **kws, forward=True)
    # mimic any()
    mask = (rolling.sum() > 0) & d.notna()

    flags[mask, field] = flag
    return data, flags


@flagging(masking="field")
def flagByVariance(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: FreqString = "12h",
    thresh: float = 0.0005,
    maxna: int = None,
    maxna_group: int = None,
    flag: float = BAD,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:

    (1) n > `window`
    (2) variance(y(t),...,y(t+n) < `thresh`

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store quality flags to data.
    window : str
        Only intervals of minimum size "window" have the chance to get flagged as constant intervals
    thresh : float
        The upper bound, the variance of an interval must not exceed, if the interval wants to be flagged a plateau.
    maxna : int, default None
        Maximum number of NaNs tolerated in an interval. If more NaNs are present, the
        interval is not flagged as plateau.
    maxna_group : int, default None
        Same as `maxna` but for consecutive NaNs.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The flags object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flags input.
    """
    dataseries = data[field]
    delta = getFreqDelta(dataseries.index)
    if not delta:
        raise IndexError("Timeseries irregularly sampled!")

    if maxna is None:
        maxna = np.inf

    if maxna_group is None:
        maxna_group = np.inf

    min_periods = int(np.ceil(pd.Timedelta(window) / pd.Timedelta(delta)))
    window = pd.Timedelta(window)
    to_set = statPass(
        dataseries,
        lambda x: varQC(x, maxna, maxna_group),
        window,
        thresh,
        operator.lt,
        min_periods=min_periods,
    )

    flags[to_set, field] = flag
    return data, flags
