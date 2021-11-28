#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Detecting breaks in data.

This module provides functions to detect and flag breaks in data, for example temporal
gaps (:py:func:`flagMissing`), jumps and drops (:py:func:`flagJumps`) or temporal
isolated values (:py:func:`flagIsolated`).
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
    Flag NaNs in data.

    By default only NaNs are flagged, that not already have a flag.
    `to_mask` can be used to pass a flag that is used as threshold.
    Each flag worse than the threshold is replaced by the function.
    This is, because the data gets masked (with NaNs) before the
    function evaluates the NaNs.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        Column(s) in flags and data.

    flags : saqc.Flags
        The flags container.

    flag : float, default BAD
        Flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        Unmodified data container
    flags : saqc.Flags
        The flags container
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
    Find and flag temporal isolated groups of data.

    The function flags arbitrarily large groups of values, if they are surrounded by
    sufficiently large data gaps. A gap is a timespan containing either no data at all
    or NaNs only.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        Column(s) in flags and data.

    flags : saqc.Flags
        The flags container.

    gap_window : str
        Minimum gap size required before and after a data group to consider it
        isolated. See condition (2) and (3)

    group_window : str
        Maximum size of a data chunk to consider it a candidate for an isolated group.
        Data chunks that are bigger than the ``group_window`` are ignored.
        This does not include the possible gaps surrounding it.
        See condition (1).

    flag : float, default BAD
        Flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        Unmodified data container
    flags : saqc.Flags
        The flags container

    Notes
    -----
    A series of values :math:`x_k,x_{k+1},...,x_{k+n}`, with associated
    timestamps :math:`t_k,t_{k+1},...,t_{k+n}`, is considered to be isolated, if:

    1. :math:`t_{k+1} - t_n <` `group_window`
    2. None of the :math:`x_j` with :math:`0 < t_k - t_j <` `gap_window`,
        is valid (preceeding gap).
    3. None of the :math:`x_j` with :math:`0 < t_j - t_(k+n) <` `gap_window`,
        is valid (succeding gap).
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
    Flag jumps and drops in data.

    Flag data where the mean of its values significantly changes (the data "jumps").

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        Column(s) in flags and data.

    flags : saqc.Flags
        The flags container.

    thresh : float
        Threshold value by which the mean of data has to change to trigger flagging.

    window : str
        Size of the moving window. This is the number of observations used
        for calculating the statistic.

    min_periods : int, default 1
        Minimum number of observations in window required to calculate a valid
        mean value.

    flag : float, default BAD
        Flag to set.
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
