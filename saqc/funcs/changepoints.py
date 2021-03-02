#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pandas as pd
import numpy as np
import numba
from typing import Callable, Union, Tuple, Optional
from typing_extensions import Literal

from dios import DictOfSeries

from saqc.common import *
from saqc.core.register import register
from saqc.lib.tools import customRoller
from saqc.flagger import Flagger
from saqc.lib.types import ColumnName, FreqString, IntegerWindow

logger = logging.getLogger("SaQC")


@register(masking='field', module="changepoints")
def flagChangePoints(
        data: DictOfSeries, field: str, flagger: Flagger,
        stat_func: Callable[[np.ndarray, np.ndarray], float],
        thresh_func: Callable[[np.ndarray, np.ndarray], float],
        bwd_window: FreqString,
        min_periods_bwd: IntegerWindow,
        fwd_window: Optional[FreqString]=None,
        min_periods_fwd: Optional[IntegerWindow]=None,
        closed: Literal["right", "left", "both", "neither"]="both",
        try_to_jit: bool=True,
        reduce_window: FreqString=None,
        reduce_func: Callable[[np.ndarray, np.ndarray], int]=lambda x, _: x.argmax(),
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
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
    fwd_window : {None, str}, default None
        The right (forward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, str, int}, default None
        Minimum number of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.
    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.
    reduce_func : Callable[[numpy.ndarray, numpy.ndarray], int], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.


    Returns
    -------

    """

    data, flagger = assignChangePointCluster(
        data, field, flagger, stat_func=stat_func, thresh_func=thresh_func,
        bwd_window=bwd_window, min_periods_bwd=min_periods_bwd,
        fwd_window=fwd_window, min_periods_fwd=min_periods_fwd, closed=closed,
        try_to_jit=try_to_jit, reduce_window=reduce_window,
        reduce_func=reduce_func, flag_changepoints=True, model_by_resids=False,
        assign_cluster=False, **kwargs
    )
    return data, flagger


@register(masking='field', module="changepoints")
def assignChangePointCluster(
        data: DictOfSeries, field: str, flagger: Flagger,
        stat_func: Callable[[np.array, np.array], float],
        thresh_func: Callable[[np.array, np.array], float],
        bwd_window: str,
        min_periods_bwd: int,
        fwd_window: str=None,
        min_periods_fwd: Optional[int]=None,
        closed: Literal["right", "left", "both", "neither"]="both",
        try_to_jit: bool=True,
        reduce_window: str=None,
        reduce_func: Callable[[np.ndarray, np.ndarray], float]=lambda x, _: x.argmax(),
        model_by_resids: bool=False,
        flag_changepoints: bool=False,
        assign_cluster: bool=True,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:

    """
    Assigns label to the data, aiming to reflect continous regimes of the processes the data is assumed to be
    generated by.
    The regime change points detection is based on a sliding window search.

    Note, that the cluster labels will be stored to the `field` field of the input data, so that the data that is
    clustered gets overridden.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The reference variable, the deviation from wich determines the flagging.
    flagger : saqc.flagger
        A flagger object, holding flags and additional informations related to `data`.
    stat_func : Callable[[numpy.array, numpy.array], float]
        A function that assigns a value to every twin window. Left window content will be passed to first variable,
        right window content will be passed to the second.
    thresh_func : Callable[numpy.array, numpy.array], float]
        A function that determines the value level, exceeding wich qualifies a timestamps stat func value as denoting a
        changepoint.
    bwd_window : str
        The left (backwards facing) windows temporal extension (freq-string).
    min_periods_bwd : int
        Minimum number of periods that have to be present in a backwards facing window, for a changepoint test to be
        performed.
    fwd_window : {None, str}, default None
        The right (forward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, int}, default None
        Minimum number of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.
    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.
    reduce_func : Callable[[numpy.array, numpy.array], numpy.array], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.
    flag_changepoints : bool, default False
        If true, the points, where there is a change in data modelling regime detected get flagged bad.
    model_by_resids : bool, default False
        If True, the data is replaced by the stat_funcs results instead of regime labels.
    assign_cluster : bool, default True
        Is set to False, if called by function that oly wants to calculate flags.

    Returns
    -------

    """
    data = data.copy()
    data_ser = data[field].dropna()
    if fwd_window is None:
        fwd_window = bwd_window
    if min_periods_fwd is None:
        min_periods_fwd = min_periods_bwd
    if reduce_window is None:
        reduce_window = f"{int(pd.Timedelta(bwd_window).total_seconds() + pd.Timedelta(fwd_window).total_seconds())}s"

    roller = customRoller(data_ser, window=bwd_window)
    bwd_start, bwd_end = roller.window.get_window_bounds(len(data_ser), min_periods=min_periods_bwd, closed=closed)

    roller = customRoller(data_ser, window=fwd_window, forward=True)
    fwd_start, fwd_end = roller.window.get_window_bounds(len(data_ser), min_periods=min_periods_fwd, closed=closed)

    min_mask = ~((fwd_end - fwd_start <= min_periods_fwd) | (bwd_end - bwd_start <= min_periods_bwd))
    fwd_end = fwd_end[min_mask]
    split = bwd_end[min_mask]
    bwd_start = bwd_start[min_mask]
    masked_index = data_ser.index[min_mask]
    check_len = len(fwd_end)
    data_arr = data_ser.values

    if try_to_jit:
        jit_sf = numba.jit(stat_func, nopython=True)
        jit_tf = numba.jit(thresh_func, nopython=True)
        try:
            jit_sf(data_arr[bwd_start[0]:bwd_end[0]], data_arr[fwd_start[0]:fwd_end[0]])
            jit_tf(data_arr[bwd_start[0]:bwd_end[0]], data_arr[fwd_start[0]:fwd_end[0]])
            stat_func = jit_sf
            thresh_func = jit_tf
            try_to_jit = True
        except (numba.core.errors.TypingError, numba.core.errors.UnsupportedError, IndexError):
            try_to_jit = False
            logging.warning('Could not jit passed statistic - omitting jitting!')

    if try_to_jit:
        stat_arr, thresh_arr = _slidingWindowSearchNumba(data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, check_len)
    else:
        stat_arr, thresh_arr = _slidingWindowSearch(data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, check_len)
    result_arr = stat_arr > thresh_arr

    if model_by_resids:
        residues = pd.Series(np.nan, index=data[field].index)
        residues[masked_index] = stat_arr
        data[field] = residues
        flagger = flagger.setFlags(field, flag=UNFLAGGED, force=True, **kwargs)
        return data, flagger

    det_index = masked_index[result_arr]
    detected = pd.Series(True, index=det_index)
    if reduce_window:
        l = detected.shape[0]
        roller = customRoller(detected, window=reduce_window)
        start, end = roller.window.get_window_bounds(num_values=l, min_periods=1, closed='both', center=True)

        detected = _reduceCPCluster(stat_arr[result_arr], thresh_arr[result_arr], start, end, reduce_func, l)
        det_index = det_index[detected]

    if assign_cluster:
        cluster = pd.Series(False, index=data[field].index)
        cluster[det_index] = True
        cluster = cluster.cumsum()
        # (better to start cluster labels with number one)
        cluster += 1
        data[field] = cluster
        flagger = flagger.setFlags(field, flag=UNFLAGGED, force=True, **kwargs)

    if flag_changepoints:
        flagger = flagger.setFlags(field, loc=det_index)
    return data, flagger


@numba.jit(parallel=True, nopython=True)
def _slidingWindowSearchNumba(data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, num_val):
    stat_arr = np.zeros(num_val)
    thresh_arr = np.zeros(num_val)
    for win_i in numba.prange(0, num_val-1):
        x = data_arr[bwd_start[win_i]:split[win_i]]
        y = data_arr[split[win_i]:fwd_end[win_i]]
        stat_arr[win_i] = stat_func(x, y)
        thresh_arr[win_i] = thresh_func(x, y)
    return stat_arr, thresh_arr


def _slidingWindowSearch(data_arr, bwd_start, fwd_end, split, stat_func, thresh_func, num_val):
    stat_arr = np.zeros(num_val)
    thresh_arr = np.zeros(num_val)
    for win_i in range(0, num_val-1):
        x = data_arr[bwd_start[win_i]:split[win_i]]
        y = data_arr[split[win_i]:fwd_end[win_i]]
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
