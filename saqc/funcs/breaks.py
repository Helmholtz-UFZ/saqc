#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Detecting breakish changes in timeseries value courses.

This module provides functions to detect and flag  breakish changes in the data value course, like gaps
(:py:func:`flagMissing`), jumps/drops (:py:func:`flagJumps`) or isolated values (:py:func:`flagIsolated`).
"""

from typing import Tuple

import numpy as np
import pandas as pd
import pandas.tseries.frequencies

from dios import DictOfSeries

from saqc.constants import *
from saqc.lib.tools import groupConsecutives
from saqc.funcs.changepoints import _assignChangePointCluster
from saqc.core.flags import Flags
from saqc.core.history import History
from saqc.core.register import _isflagged, register, flagging


@register(mask=[], demask=[], squeeze=["field"])
def flagMissing(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    flag: float = BAD,
    dfilter: float = FILTER_ALL,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function flags all values indicating missing data.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store quality flags to data.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
    """

    datacol = data[field]
    mask = datacol.isna()

    mask = ~_isflagged(flags[field], dfilter) & mask

    flags[mask, field] = flag
    return data, flags


@flagging()
def flagIsolated(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    gap_window: str,
    group_window: str,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function flags arbitrary large groups of values, if they are surrounded by sufficiently
    large data gaps.

    A gap is a timespan containing either no data or data invalid only (usually `nan`) .

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        A flags object
    gap_window : str
        The minimum size of the gap before and after a group of valid values, making this group considered an
        isolated group. See condition (2) and (3)
    group_window : str
        The maximum temporal extension allowed for a group that is isolated by gaps of size 'gap_window',
        to be actually flagged as isolated group. See condition (1).
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The flags object, holding flags and additional information related to `data`.

    Notes
    -----
    A series of values :math:`x_k,x_{k+1},...,x_{k+n}`, with associated timestamps :math:`t_k,t_{k+1},...,t_{k+n}`,
    is considered to be isolated, if:

    1. :math:`t_{k+1} - t_n <` `group_window`
    2. None of the :math:`x_j` with :math:`0 < t_k - t_j <` `gap_window`, is valid (preceeding gap).
    3. None of the :math:`x_j` with :math:`0 < t_j - t_(k+n) <` `gap_window`, is valid (succeding gap).

    See Also
    --------
    :py:func:`flagMissing`
    """
    gap_window = pd.tseries.frequencies.to_offset(gap_window)
    group_window = pd.tseries.frequencies.to_offset(group_window)

    mask = data[field].isna()

    bools = pd.Series(data=0, index=mask.index, dtype=bool)
    for srs in groupConsecutives(mask):
        if np.all(~srs):
            # we found a chunk of non-nan values
            start = srs.index[0]
            stop = srs.index[-1]
            if stop - start <= group_window:
                # the chunk is large enough
                left = mask[start - gap_window : start].iloc[:-1]
                if left.all():
                    # the section before our chunk is nan-only
                    right = mask[stop : stop + gap_window].iloc[1:]
                    if right.all():
                        # the section after our chunk is nan-only
                        # -> we found a chunk of isolated non-values
                        bools[start:stop] = True

    flags[bools, field] = flag
    return data, flags


@flagging()
def flagJumps(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    thresh: float,
    window: str,
    min_periods: int = 1,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Flag where the mean of the values significantly changes (the data "jumps").

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.
    thresh : float
        The threshold, the mean of the values have to change by, to trigger flagging.
    window : str
        The temporal extension, of the rolling windows, the mean values that are to be
        compared, are obtained from.
    min_periods : int, default 1
        Minimum number of periods that have to be present in a window of size `window`,
        so that the mean value obtained from that window is regarded valid.
    flag : float, default BAD
        flag to set.
    """
    return _assignChangePointCluster(
        data,
        field,
        flags,
        stat_func=lambda x, y: np.abs(np.mean(x) - np.mean(y)),
        thresh_func=lambda x, y: thresh,
        window=window,
        min_periods=min_periods,
        set_flags=True,
        model_by_resids=False,
        assign_cluster=False,
        flag=flag,
        **kwargs,
    )
