#! /usr/bin/env python
# -*- coding: utf-8 -*-


import dios
import numpy as np
import pandas as pd


from saqc.lib.tools import groupConsecutives
from saqc.funcs.changepoints import assignChangePointCluster
from saqc.core.register import register


@register(masking='field')
def flagMissing(data, field, flagger, nodata=np.nan, **kwargs):
    """
    The function flags all values indicating missing data.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    nodata : any, default np.nan
        A value that defines missing data.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    datacol = data[field]
    if np.isnan(nodata):
        mask = datacol.isna()
    else:
        mask = datacol == nodata

    flagger = flagger.setFlags(field, loc=mask, **kwargs)
    return data, flagger


@register(masking='field')
def flagIsolated(data, field, flagger, gap_window, group_window, **kwargs):
    """
    The function flags arbitrary large groups of values, if they are surrounded by sufficiently
    large data gaps. A gap is defined as group of missing and/or flagged values.

    A series of values x_k,x_(k+1),...,x_(k+n), with associated timestamps t_k,t_(k+1),...,t_(k+n),
    is considered to be isolated, if:

    1. t_(k+1) - t_n < `group_window`
    2. None of the x_j with 0 < t_k - t_j < `gap_window`, is valid or unflagged (preceeding gap).
    3. None of the x_j with 0 < t_j - t_(k+n) < `gap_window`, is valid or unflagged (succeding gap).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    gap_window :
        The minimum size of the gap before and after a group of valid values, making this group considered an
        isolated group. See condition (2) and (3)
    group_window :
        The maximum temporal extension allowed for a group that is isolated by gaps of size 'gap_window',
        to be actually flagged as isolated group. See condition (1).

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    gap_window = pd.tseries.frequencies.to_offset(gap_window)
    group_window = pd.tseries.frequencies.to_offset(group_window)

    col = data[field].mask(flagger.isFlagged(field))
    mask = col.isnull()

    flags = pd.Series(data=0, index=col.index, dtype=bool)
    for srs in groupConsecutives(mask):
        if np.all(~srs):
            start = srs.index[0]
            stop = srs.index[-1]
            if stop - start <= group_window:
                left = mask[start - gap_window: start].iloc[:-1]
                if left.all():
                    right = mask[stop: stop + gap_window].iloc[1:]
                    if right.all():
                        flags[start:stop] = True

    flagger = flagger.setFlags(field, flags, **kwargs)

    return data, flagger


@register(masking='field')
def flagJumps(data, field, flagger, tresh, winsz, min_periods=1, **kwargs):
    """,
    Flag datapoints, where the parametrization of the process, the data is assumed to generate by, significantly
    changes.

    The change points detection is based on a sliding window search.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flagger : saqc.flagger
        A flagger object, holding flags and additional informations related to `data`.
    stat_func : Callable[numpy.array, numpy.array]
         A function that assigns a value to every twin window. Left window content will be passed to first variable,
        right window content will be passed to the second.
    thresh_func : Callable[numpy.array, numpy.array]
        A function that determines the value level, exceeding wich qualifies a timestamps stat func value as denoting a
        changepoint.
    bwd_window : str
        The left (backwards facing) windows temporal extension (freq-string).
    min_periods_bwd : {str, int}
        Minimum number of periods that have to be present in a backwards facing window, for a changepoint test to be
        performed.
    fwd_window : {Non/home/luenensc/PyPojects/testSpace/flagBasicMystery.pye, str}, default None
        The right (fo/home/luenensc/PyPojects/testSpace/flagBasicMystery.pyrward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, str, int}, default None
        Minimum numbe/home/luenensc/PyPojects/testSpace/flagBasicMystery.pyr of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.
    reduce_window : {None, False, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is not False, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.
    reduce_func : Callable[numpy.array, numpy.array], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.


    Returns
    -------

    """

    data, flagger = assignChangePointCluster(data, field, flagger,
                                             stat_func=lambda x, y: np.abs(np.mean(x) - np.mean(y)),
                                             tresh_func=lambda x, y: tresh,
                                             bwd_window=winsz,
                                             min_periods_bwd=min_periods,
                                             flag_changepoints=True,
                                             _model_by_resids=False,
                                             _assign_cluster=False)

    return data, flagger