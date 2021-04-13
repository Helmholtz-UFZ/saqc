#! /usr/bin/env python
# -*- coding: utf-8 -*-

from operator import mod
from typing import Tuple

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.constants import *
from saqc.core import register, Flags
from saqc.lib.ts_operators import varQC
from saqc.lib.tools import customRoller, getFreqDelta
from saqc.lib.types import FreqString, ColumnName


@register(masking="field", module="constants")
def flagConstants(
    data: DictOfSeries,
    field: ColumnName,
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

    # find all consecutive constant values in one direction...
    r = customRoller(d, **kws)
    m1 = r.max() - r.min() <= thresh
    # and in the other
    r = customRoller(d, forward=True, **kws)
    m2 = r.max() - r.min() <= thresh
    mask = m1 | m2

    flags[mask, field] = flag
    return data, flags


@register(masking="field", module="constants")
def flagByVariance(
    data: DictOfSeries,
    field: ColumnName,
    flags: Flags,
    window: FreqString = "12h",
    thresh: float = 0.0005,
    max_missing: int = None,
    max_consec_missing: int = None,
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
    max_missing : {None, int}, default None
        Maximum number of nan values tolerated in an interval, for retrieving a valid
        variance from it. (Intervals with a number of nans exceeding "max_missing"
        have no chance to get flagged a plateau!)
    max_consec_missing : {None, int}, default None
        Maximum number of consecutive nan values allowed in an interval to retrieve a
        valid  variance from it. (Intervals with a number of nans exceeding
        "max_consec_missing" have no chance to get flagged a plateau!)
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

    if max_missing is None:
        max_missing = np.inf

    if max_consec_missing is None:
        max_consec_missing = np.inf

    min_periods = int(np.ceil(pd.Timedelta(window) / pd.Timedelta(delta)))

    def var_below_thresh(s: pd.Series):
        if varQC(s, max_missing, max_consec_missing) <= thresh:
            return True
        return np.nan

    rolling = dataseries.rolling(window=window, min_periods=min_periods)
    plateaus = rolling.apply(var_below_thresh, raw=False)

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flags

    plateaus.fillna(method="bfill", limit=min_periods - 1, inplace=True)

    # result:
    plateaus = (plateaus[plateaus == 1.0]).index

    flags[plateaus, field] = flag
    return data, flags
