#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
import numba

from typing import Callable, Tuple
from typing_extensions import Literal

from dios import DictOfSeries

from saqc.constants import *
from saqc.core.register import flagging
from saqc.lib.tools import customRoller, filterKwargs
from saqc.core import register, Flags


@flagging()
def flagChangePoints(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    stat_func: Callable[[np.ndarray, np.ndarray], float],
    thresh_func: Callable[[np.ndarray, np.ndarray], float],
    window: str | Tuple[str, str],
    min_periods: int | Tuple[int, int],
    closed: Literal["right", "left", "both", "neither"] = "both",
    reduce_window: str = None,
    reduce_func: Callable[[np.ndarray, np.ndarray], int] = lambda x, _: x.argmax(),
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Flag data where it significantly changes.

    Flag data points, where the parametrization of the process, the data is assumed to
    generate by, significantly changes.

    The change points detection is based on a sliding window search.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        A column in flags and data.

    flags : saqc.Flags
        The flags container.

    stat_func : Callable
         A function that assigns a value to every twin window. The backward-facing
         window content will be passed as the first array, the forward-facing window
         content as the second.

    thresh_func : Callable
        A function that determines the value level, exceeding wich qualifies a
        timestamps func value as denoting a change-point.

    window : str, tuple of str
        Size of the moving windows. This is the number of observations used for
        calculating the statistic.

        If it is a single frequency offset, it applies for the backward- and the
        forward-facing window.

        If two offsets (as a tuple) is passed the first defines the size of the
        backward facing window, the second the size of the forward facing window.

    min_periods : int or tuple of int
        Minimum number of observations in a window required to perform the changepoint
        test. If it is a tuple of two int, the first refer to the backward-,
        the second to the forward-facing window.

    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.

    reduce_window : str or None, default None
        The sliding window search method is not an exact CP search method and usually
        there wont be detected a single changepoint, but a "region" of change around
        a changepoint.

        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will
        be dropped.

        If `reduce_window` is None, the reduction window size equals the twin window
        size, the changepoints have been detected with.

    reduce_func : Callable, default ``lambda x, y: x.argmax()``
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for
        every reduction window. Second input parameter holds the result from the
        `thresh_func` evaluation.
        The default reduction function just selects the value that maximizes the
        `stat_func`.

    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        Unmodified data container
    flags : saqc.Flags
        The flags container
    """
    return _assignChangePointCluster(
        data,
        field,
        flags,
        stat_func=stat_func,
        thresh_func=thresh_func,
        window=window,
        min_periods=min_periods,
        closed=closed,
        reduce_window=reduce_window,
        reduce_func=reduce_func,
        set_flags=True,
        model_by_resids=False,
        assign_cluster=False,
        flag=flag,
        **kwargs,
    )


@register(mask=["field"], demask=[], squeeze=[])
def assignChangePointCluster(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    stat_func: Callable[[np.array, np.array], float],
    thresh_func: Callable[[np.array, np.array], float],
    window: str | Tuple[str, str],
    min_periods: int | Tuple[int, int],
    closed: Literal["right", "left", "both", "neither"] = "both",
    reduce_window: str = None,
    reduce_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, _: x.argmax(),
    model_by_resids: bool = False,
    **kwargs,
):
    """
    Label data where it changes significantly.

    The labels will be stored in data. Unless `target` is given the labels will
    overwrite the data in `field`. The flags will always set to `UNFLAGGED`.

    Assigns label to the data, aiming to reflect continuous regimes of the processes
    the data is assumed to be generated by. The regime change points detection is
    based on a sliding window search.


    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The reference variable, the deviation from wich determines the flagging.

    flags : saqc.flags
        A flags object, holding flags and additional informations related to `data`.

    stat_func : Callable[[numpy.array, numpy.array], float]
        A function that assigns a value to every twin window. Left window content will
        be passed to first variable,
        right window content will be passed to the second.

    thresh_func : Callable[numpy.array, numpy.array], float]
        A function that determines the value level, exceeding wich qualifies a
        timestamps func func value as denoting a changepoint.

    window : str, tuple of string
        Size of the rolling windows the calculation is performed in. If it is a single
        frequency offset, it applies for the backward- and the forward-facing window.

        If two offsets (as a tuple) is passed the first defines the size of the
        backward facing window, the second the size of the forward facing window.

    min_periods : int or tuple of int
        Minimum number of observations in a window required to perform the changepoint
        test. If it is a tuple of two int, the first refer to the backward-,
        the second to the forward-facing window.

    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.

    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually
        there wont be detected a single changepoint, but a "region" of change around
        a changepoint. If `reduce_window` is given, for every window of size
        `reduce_window`, there will be selected the value with index `reduce_func(x,
        y)` and the others will be dropped. If `reduce_window` is None, the reduction
        window size equals the twin window size, the changepoints have been detected
        with.

    reduce_func : callable, default lambda x,y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for
        every reduction window. Second input parameter holds the result from the
        thresh_func evaluation. The default reduction function just selects the value
        that maximizes the stat_func.

    model_by_resids : bool, default False
        If True, the results of `stat_funcs` are written, otherwise the regime labels.

    Returns
    -------
    data : dios.DictOfSeries
        Modified data.
    flags : saqc.Flags
        The flags container
    """
    reserved = ["assign_cluster", "set_flags", "flag"]
    kwargs = filterKwargs(kwargs, reserved)
    return _assignChangePointCluster(
        data=data,
        field=field,
        flags=flags,
        stat_func=stat_func,
        thresh_func=thresh_func,
        window=window,
        min_periods=min_periods,
        closed=closed,
        reduce_window=reduce_window,
        reduce_func=reduce_func,
        model_by_resids=model_by_resids,
        **kwargs,
        # control args
        assign_cluster=True,
        set_flags=False,
    )


def _assignChangePointCluster(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    stat_func: Callable[[np.array, np.array], float],
    thresh_func: Callable[[np.array, np.array], float],
    window: str | Tuple[str, str],
    min_periods: int | Tuple[int, int],
    closed: Literal["right", "left", "both", "neither"] = "both",
    reduce_window: str = None,
    reduce_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, _: x.argmax(),
    model_by_resids: bool = False,
    set_flags: bool = False,
    assign_cluster: bool = True,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    data_ser = data[field].dropna()
    if isinstance(window, (list, tuple)):
        bwd_window, fwd_window = window
    else:
        bwd_window = fwd_window = window

    if isinstance(window, (list, tuple)):
        bwd_min_periods, fwd_min_periods = min_periods
    else:
        bwd_min_periods = fwd_min_periods = min_periods

    if reduce_window is None:
        s = int(
            pd.Timedelta(bwd_window).total_seconds()
            + pd.Timedelta(fwd_window).total_seconds()
        )
        reduce_window = f"{s}s"

    roller = customRoller(data_ser, window=bwd_window, min_periods=bwd_min_periods)
    bwd_start, bwd_end = roller.window_indexer.get_window_bounds(
        len(data_ser), min_periods=bwd_min_periods, closed=closed
    )

    roller = customRoller(
        data_ser, window=fwd_window, forward=True, min_periods=fwd_min_periods
    )
    fwd_start, fwd_end = roller.window_indexer.get_window_bounds(
        len(data_ser), min_periods=fwd_min_periods, closed=closed
    )

    min_mask = ~(
        (fwd_end - fwd_start <= fwd_min_periods)
        | (bwd_end - bwd_start <= bwd_min_periods)
    )
    fwd_end = fwd_end[min_mask]
    split = bwd_end[min_mask]
    bwd_start = bwd_start[min_mask]
    masked_index = data_ser.index[min_mask]
    check_len = len(fwd_end)
    data_arr = data_ser.values

    try_to_jit = True
    jit_sf = numba.jit(stat_func, nopython=True)
    jit_tf = numba.jit(thresh_func, nopython=True)
    try:
        jit_sf(data_arr[bwd_start[0] : bwd_end[0]], data_arr[fwd_start[0] : fwd_end[0]])
        jit_tf(data_arr[bwd_start[0] : bwd_end[0]], data_arr[fwd_start[0] : fwd_end[0]])
        stat_func = jit_sf
        thresh_func = jit_tf
    except (numba.TypingError, numba.UnsupportedError, IndexError):
        try_to_jit = False

    args = data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, check_len

    if try_to_jit:
        stat_arr, thresh_arr = _slidingWindowSearchNumba(*args)
    else:
        stat_arr, thresh_arr = _slidingWindowSearch(*args)

    result_arr = stat_arr > thresh_arr

    if model_by_resids:
        residues = pd.Series(np.nan, index=data[field].index)
        residues[masked_index] = stat_arr
        data[field] = residues
        flags[:, field] = UNFLAGGED
        return data, flags

    det_index = masked_index[result_arr]
    detected = pd.Series(True, index=det_index)
    if reduce_window:
        l = detected.shape[0]
        roller = customRoller(detected, window=reduce_window, min_periods=1)
        start, end = roller.window_indexer.get_window_bounds(
            num_values=l, min_periods=1, closed="both", center=True
        )

        detected = _reduceCPCluster(
            stat_arr[result_arr], thresh_arr[result_arr], start, end, reduce_func, l
        )
        det_index = det_index[detected]

    if assign_cluster:
        cluster = pd.Series(False, index=data[field].index)
        cluster[det_index] = True
        cluster = cluster.cumsum()
        # (better to start cluster labels with number one)
        cluster += 1
        data[field] = cluster
        flags[:, field] = UNFLAGGED

    if set_flags:
        flags[det_index, field] = flag
    return data, flags


@numba.jit(parallel=True, nopython=True)
def _slidingWindowSearchNumba(
    data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, num_val
):
    stat_arr = np.zeros(num_val)
    thresh_arr = np.zeros(num_val)
    for win_i in numba.prange(0, num_val - 1):
        x = data_arr[bwd_start[win_i] : split[win_i]]
        y = data_arr[split[win_i] : fwd_end[win_i]]
        stat_arr[win_i] = stat_func(x, y)
        thresh_arr[win_i] = thresh_func(x, y)
    return stat_arr, thresh_arr


def _slidingWindowSearch(
    data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, num_val
):
    stat_arr = np.zeros(num_val)
    thresh_arr = np.zeros(num_val)
    for win_i in range(0, num_val - 1):
        x = data_arr[bwd_start[win_i] : split[win_i]]
        y = data_arr[split[win_i] : fwd_end[win_i]]
        stat_arr[win_i] = stat_func(x, y)
        thresh_arr[win_i] = thresh_func(x, y)
    return stat_arr, thresh_arr


def _reduceCPCluster(stat_arr, thresh_arr, start, end, obj_func, num_val):
    out_arr = np.zeros(shape=num_val, dtype=bool)
    for win_i in numba.prange(0, num_val):
        s, e = start[win_i], end[win_i]
        x = stat_arr[s:e]
        y = thresh_arr[s:e]
        pos = s + obj_func(x, y) + 1
        out_arr[s:e] = False
        out_arr[pos] = True

    return out_arr
