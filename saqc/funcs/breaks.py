#! /usr/bin/env python
# -*- coding: utf-8 -*-


from dios import DictOfSeries
import numpy as np
import pandas as pd
from typing import Tuple


from saqc.lib.tools import groupConsecutives
from saqc.funcs.changepoints import assignChangePointCluster
from saqc.core.register import register
from saqc.flagger.baseflagger import BaseFlagger


@register(masking='field')
def flagMissing(data: DictOfSeries, field: str, flagger: BaseFlagger, nodata: float=np.nan, **kwargs) -> Tuple[DictOfSeries, BaseFlagger]:
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
def flagIsolated(data: DictOfSeries, field: str, flagger: BaseFlagger, gap_window: str, group_window: str, **kwargs) -> Tuple[DictOfSeries, BaseFlagger]:
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
    gap_window : str
        The minimum size of the gap before and after a group of valid values, making this group considered an
        isolated group. See condition (2) and (3)
    group_window : str
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
def flagJumps(data: DictOfSeries, field: str, flagger: BaseFlagger, thresh: float, winsz: str, min_periods: int=1,
              **kwargs):
    """
    Flag datapoints, where the mean of the values significantly changes (where the value course "jumps").

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flagger : saqc.flagger
        A flagger object, holding flags and additional informations related to `data`.
    thresh : float
        The threshold, the mean of the values have to change by, to trigger flagging.
    winsz : str
        The temporal extension, of the rolling windows, the mean values that are to be compared,
        are obtained from.
    min_periods : int, default 1
        Minimum number of periods that have to be present in a window of size `winsz`, so that
        the mean value obtained from that window is regarded valid.
    """

    data, flagger = assignChangePointCluster(data, field, flagger,
                                             stat_func=lambda x, y: np.abs(np.mean(x) - np.mean(y)),
                                             thresh_func=lambda x, y: thresh,
                                             bwd_window=winsz,
                                             min_periods_bwd=min_periods,
                                             flag_changepoints=True,
                                             model_by_resids=False,
                                             assign_cluster=False,
                                             **kwargs)

    return data, flagger
