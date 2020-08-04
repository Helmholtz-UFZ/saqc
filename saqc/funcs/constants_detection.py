#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from saqc.core.register import register
from saqc.lib.ts_operators import varQC
from saqc.lib.tools import retrieveTrustworthyOriginal


@register
def constants_flagBasic(data, field, flagger, thresh, window, **kwargs):
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
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
    thresh : float
        Upper bound for the maximum total change of an interval to be flagged constant.
    window : str
        Lower bound for the size of an interval to be flagged constant.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.
    """

    d = data[field]

    # find all constant values in a row with a forward search
    r = d.rolling(window=window)
    mask = (r.max() - r.min() <= thresh) & (r.count() > 1)

    # backward rolling for offset windows hack
    bw = mask[::-1].copy()
    bw.index = bw.index.max() - bw.index

    # propagate the mask(!), backwards
    bwmask = bw.rolling(window=window).sum() > 0

    mask |= bwmask[::-1].values

    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register
def constants_flagVarianceBased(
    data, field, flagger, window="12h", thresh=0.0005, max_missing=None, max_consec_missing=None, **kwargs
):

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
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
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

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.
    """

    dataseries, data_rate = retrieveTrustworthyOriginal(data, field, flagger)

    if max_missing is None:
        max_missing = np.inf
    if max_consec_missing is None:
        max_consec_missing = np.inf

    min_periods = int(np.ceil(pd.Timedelta(window) / pd.Timedelta(data_rate)))

    plateaus = dataseries.rolling(window=window, min_periods=min_periods).apply(
        lambda x: True if varQC(x, max_missing, max_consec_missing) <= thresh else np.nan, raw=False,
    )

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flagger

    plateaus.fillna(method="bfill", limit=min_periods - 1, inplace=True)

    # result:
    plateaus = (plateaus[plateaus == 1.0]).index

    flagger = flagger.setFlags(field, plateaus, **kwargs)
    return data, flagger
