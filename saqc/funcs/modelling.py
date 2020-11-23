#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numba
from saqc.core.register import register
from saqc.lib.ts_operators import (
    polyRoller,
    polyRollerNoMissing,
    polyRollerNumba,
    polyRollerNoMissingNumba,
    polyRollerIrregular,
    count
)
from saqc.lib.tools import seasonalMask, customRoller
import logging

logger = logging.getLogger("SaQC")


@register(masking='field')
def modelling_polyFit(data, field, flagger, winsz, polydeg, numba="auto", eval_flags=True, min_periods=0, **kwargs):
    """
    Function fits a polynomial model to the data and returns the residues.

    The residue for value x is calculated by fitting a polynomial of degree "polydeg" to a data slice
    of size "winsz", wich has x at its center.

    Note, that the residues will be stored to the `field` field of the input data, so that the original data, the
    polynomial is fitted to, gets overridden.

    Note, that, if data[field] is not alligned to an equidistant frequency grid, the window size passed,
    has to be an offset string. Also numba boost options don`t apply for irregularly sampled
    timeseries.

    Note, that calculating the residues tends to be quite costy, because a function fitting is perfomed for every
    sample. To improve performance, consider the following possibillities:

    In case your data is sampled at an equidistant frequency grid:

    (1) If you know your data to have no significant number of missing values, or if you do not want to
        calculate residues for windows containing missing values any way, performance can be increased by setting
        min_periods=winsz.

    (2) If your data consists of more then around 200000 samples, setting numba=True, will boost the
        calculations up to a factor of 5 (for samplesize > 300000) - however for lower sample sizes,
        numba will slow down the calculations, also, up to a factor of 5, for sample_size < 50000.
        By default (numba='auto'), numba is set to true, if the data sample size exceeds 200000.

    in case your data is not sampled at an equidistant frequency grid:

    (1) Harmonization/resampling of your data will have a noticable impact on polyfittings performance - since
        numba_boost doesnt apply for irregularly sampled data in the current implementation.

    Note, that in the current implementation, the initial and final winsz/2 values do not get fitted.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    winsz : {str, int}
        The size of the window you want to use for fitting. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension. The window will be centered around the vaule-to-be-fitted.
        For regularly sampled timeseries the period number will be casted down to an odd number if
        even.
    polydeg : int
        The degree of the polynomial used for fitting
    numba : {True, False, "auto"}, default "auto"
        Wheather or not to apply numbas just-in-time compilation onto the poly fit function. This will noticably
        increase the speed of calculation, if the sample size is sufficiently high.
        If "auto" is selected, numba compatible fit functions get applied for data consisiting of > 200000 samples.
    eval_flags : bool, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
    min_periods : {int, np.nan}, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    """
    if data[field].empty:
        return data, flagger
    data = data.copy()
    to_fit = data[field]
    flags = flagger.getFlags(field)
    if not to_fit.index.freqstr:
        if isinstance(winsz, int):
            raise NotImplementedError("Integer based window size is not supported for not-harmonized" "sample series.")
        # get interval centers
        centers = np.floor((to_fit.rolling(pd.Timedelta(winsz) / 2, closed="both", min_periods=min_periods).count()))
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        residues = to_fit.rolling(pd.Timedelta(winsz), closed="both", min_periods=min_periods).apply(
            polyRollerIrregular, args=(centers, polydeg)
        )

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = centers.rolling(winsz, closed="both").apply(center_func, raw=False).astype(int)
        temp = residues.copy()
        for k in centers_iloc.iteritems():
            residues.iloc[k[1]] = temp[k[0]]
        residues[residues.index[0] : residues.index[centers_iloc[0]]] = np.nan
        residues[residues.index[centers_iloc[-1]] : residues.index[-1]] = np.nan
    else:
        if isinstance(winsz, str):
            winsz = int(np.floor(pd.Timedelta(winsz) / pd.Timedelta(to_fit.index.freqstr)))
        if winsz % 2 == 0:
            winsz = int(winsz - 1)
        if numba == "auto":
            if to_fit.shape[0] < 200000:
                numba = False
            else:
                numba = True

        val_range = np.arange(0, winsz)
        center_index = int(np.floor(winsz / 2))
        if min_periods < winsz:
            if min_periods > 0:
                to_fit = to_fit.rolling(winsz, min_periods=min_periods, center=True).apply(
                    lambda x, y: x[y], raw=True, args=(center_index,)
                )

            # we need a missing value marker that is not nan, because nan values dont get passed by pandas rolling
            # method
            miss_marker = to_fit.min()
            miss_marker = np.floor(miss_marker - 1)
            na_mask = to_fit.isna()
            to_fit[na_mask] = miss_marker
            if numba:
                residues = to_fit.rolling(winsz).apply(
                    polyRollerNumba,
                    args=(miss_marker, val_range, center_index, polydeg),
                    raw=True,
                    engine="numba",
                    engine_kwargs={"no_python": True},
                )
                # due to a tiny bug - rolling with center=True doesnt work when using numba engine.
                residues = residues.shift(-int(center_index))
            else:
                residues = to_fit.rolling(winsz, center=True).apply(
                    polyRoller, args=(miss_marker, val_range, center_index, polydeg), raw=True
                )
            residues[na_mask] = np.nan
        else:
            # we only fit fully populated intervals:
            if numba:
                residues = to_fit.rolling(winsz).apply(
                    polyRollerNoMissingNumba,
                    args=(val_range, center_index, polydeg),
                    engine="numba",
                    engine_kwargs={"no_python": True},
                    raw=True,
                )
                # due to a tiny bug - rolling with center=True doesnt work when using numba engine.
                residues = residues.shift(-int(center_index))
            else:
                residues = to_fit.rolling(winsz, center=True).apply(
                    polyRollerNoMissing, args=(val_range, center_index, polydeg), raw=True
                )

    residues = residues - to_fit
    data[field] = residues
    if eval_flags:
        num_cats, codes = flags.factorize()
        num_cats = pd.Series(num_cats, index=flags.index).rolling(winsz, center=True, min_periods=min_periods).max()
        nan_samples = num_cats[num_cats.isna()]
        num_cats.drop(nan_samples.index, inplace=True)
        to_flag = pd.Series(codes[num_cats.astype(int)], index=num_cats.index)
        to_flag = to_flag.align(nan_samples)[0]
        to_flag[nan_samples.index] = flags[nan_samples.index]
        flagger = flagger.setFlags(field, to_flag.values, **kwargs)

    return data, flagger


@register(masking='field')
def modelling_rollingMean(data, field, flagger, winsz, eval_flags=True, min_periods=0, center=True, **kwargs):
    """
    Models the data with the rolling mean and returns the residues.

    Note, that the residues will be stored to the `field` field of the input data, so that the data that is modelled
    gets overridden.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    winsz : {int, str}
        The size of the window you want to roll with. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension.
        For regularly sampled timeseries, the period number will be casted down to an odd number if
        center = True.
    eval_flags : bool, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
        Currently not implemented in combination with not-harmonized timeseries.
    min_periods : int, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the mean
        fitting to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present.
    center : bool, default True
        Wheather or not to center the window the mean is calculated of around the reference value. If False,
        the reference value is placed to the right of the window (classic rolling mean with lag.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    data = data.copy()
    to_fit = data[field]
    flags = flagger.getFlags(field)
    if to_fit.empty:
        return data, flagger

    # starting with the annoying case: finding the rolling interval centers of not-harmonized input time series:
    if (to_fit.index.freqstr is None) and center:
        if isinstance(winsz, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                'sample series when rolling with "center=True".'
            )
        # get interval centers
        centers = np.floor((to_fit.rolling(pd.Timedelta(winsz) / 2, closed="both", min_periods=min_periods).count()))
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        means = to_fit.rolling(pd.Timedelta(winsz), closed="both", min_periods=min_periods).mean()

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = centers.rolling(winsz, closed="both").apply(center_func, raw=False).astype(int)
        temp = means.copy()
        for k in centers_iloc.iteritems():
            means.iloc[k[1]] = temp[k[0]]
        # last values are false, due to structural reasons:
        means[means.index[centers_iloc[-1]] : means.index[-1]] = np.nan

    # everything is more easy if data[field] is harmonized:
    else:
        if isinstance(winsz, str):
            winsz = int(np.floor(pd.Timedelta(winsz) / pd.Timedelta(to_fit.index.freqstr)))
        if (winsz % 2 == 0) & center:
            winsz = int(winsz - 1)

        means = to_fit.rolling(window=winsz, center=center, closed="both").mean()

    residues = means - to_fit
    data[field] = residues
    if eval_flags:
        num_cats, codes = flags.factorize()
        num_cats = pd.Series(num_cats, index=flags.index).rolling(winsz, center=True, min_periods=min_periods).max()
        nan_samples = num_cats[num_cats.isna()]
        num_cats.drop(nan_samples.index, inplace=True)
        to_flag = pd.Series(codes[num_cats.astype(int)], index=num_cats.index)
        to_flag = to_flag.align(nan_samples)[0]
        to_flag[nan_samples.index] = flags[nan_samples.index]
        flagger = flagger.setFlags(field, to_flag.values, **kwargs)

    return data, flagger


def modelling_mask(data, field, flagger, mode, mask_var=None, season_start=None, season_end=None,
                   include_bounds=True):
    """
    This function realizes masking within saqc.

    Due to some inner saqc mechanics, it is not straight forwardly possible to exclude
    values or datachunks from flagging routines. This function replaces flags with np.nan
    value, wherever values are to get masked. Furthermore, the masked values get replaced by
    np.nan, so that they dont effect calculations.

    Here comes a recipe on how to apply a flagging function only on a masked chunk of the variable field:

    1. dublicate "field" in the input data (proc_fork)
    2. mask the dublicated data (modelling_mask)
    3. apply the tests you only want to be applied onto the masked data chunks (saqc_tests)
    4. project the flags, calculated on the dublicated and masked data onto the original field data
        (proc_projectFlags or flagGeneric)
    5. drop the dublicated data (proc_drop)

    To see an implemented example, checkout flagSeasonalRange in the saqc.functions module

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-masked.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    mode : {"seasonal", "mask_var"}
        The masking mode.
        - "seasonal": parameters "season_start", "season_end" are evaluated to generate a seasonal (periodical) mask
        - "mask_var": data[mask_var] is expected to be a boolean valued timeseries and is used as mask.
    mask_var : {None, str}, default None
        Only effective if mode == "mask_var"
        Fieldname of the column, holding the data that is to be used as mask. (must be moolean series)
        Neither the series` length nor its labels have to match data[field]`s index and length. An inner join of the
        indices will be calculated and values get masked where the values of the inner join are "True".
    season_start : {None, str}, default None
        Only effective if mode == "seasonal"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `season_end` parameter.
        See examples section below for some examples.
    season_end : {None, str}, default None
        Only effective if mode == "seasonal"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `season_end` parameter.
        See examples section below for some examples.
    include_bounds : boolean
        Wheather or not to include the mask defining bounds to the mask.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.


    Examples
    --------
    The `season_start` and `season_end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
    They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
    (mm=month, dd=day, HH=hour, MM=minute, SS=second)
    Single digit specifications have to be given with leading zeros.
    `season_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
    The highest date unit gives the period.
    For example:

    >>> season_start = "01T15:00:00"
    >>> season_end = "13T17:30:00"

    Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked

    >>> season_start = "01:00"
    >>> season_end = "04:00"

    All the values between the first and 4th minute of every hour get masked.

    >>> season_start = "01-01T00:00:00"
    >>> season_end = "01-03T00:00:00"

    Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
    include 00:00:00 at the first of march. To exclude this one, pass:

    >>> season_start = "01-01T00:00:00"
    >>> season_end = "02-28T23:59:59"

    To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
    season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:

    >>> season_start = "22:00:00"
    >>> season_end = "06:00:00"

    When inclusive_selection="season", all above examples work the same way, only that you now
    determine wich values NOT TO mask (=wich values are to constitute the "seasons").
    """
    data = data.copy()
    datcol_idx = data[field].index

    if mode == 'seasonal':
        to_mask = seasonalMask(datcol_idx, season_start, season_end, include_bounds)
    elif mode == 'mask_var':
        idx = data[mask_var].index.intersection(datcol_idx)
        to_mask = data.loc[idx, mask_var]
    else:
        raise ValueError("Keyword passed as masking mode is unknown ({})!".format(mode))

    data.aloc[to_mask, field] = np.nan
    flagger = flagger.setFlags(field, loc=to_mask, flag=np.nan, force=True)

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


@register(masking='field')
def modelling_changePointCluster(data, field, flagger, stat_func, thresh_func, bwd_window, min_periods_bwd,
                                 fwd_window=None, min_periods_fwd=None, closed='both', try_to_jit=True,
                                 reduce_window=None, reduce_func=lambda x, y: x.argmax(), flag_changepoints=False,
                                 model_by_resids=False, **kwargs):
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
        If `agg_range` is not False, for every window of size `agg_range`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size the changepoints have been detected with.
    reduce_func : Callable[numpy.array, numpy.array], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.
    flag_changepoints : bool, default False
        If true, the points, where there is a change in data modelling regime detected get flagged bad.
    model_by_resids _ bool, default False
        If True, the data is replaced by the stat_funcs results instead of regime labels.

    Returns
    -------

    """
    data = data.copy()
    data_ser = data[field].dropna()
    center = False
    var_len = data_ser.shape[0]
    if fwd_window is None:
        fwd_window = bwd_window
    if min_periods_fwd is None:
        min_periods_fwd = min_periods_bwd
    if reduce_window is None:
        reduce_window = f"{int(pd.Timedelta(bwd_window).total_seconds() + pd.Timedelta(fwd_window).total_seconds())}s"

    # native pandas.rolling also fails
    data_ser.rolling(window=bwd_window, min_periods=min_periods_bwd, closed=closed)
    roller = customRoller(data_ser, window=bwd_window, min_periods=min_periods_bwd, closed=closed)
    bwd_start, bwd_end = roller.window.get_window_bounds()

    roller = customRoller(data_ser, window=fwd_window, min_periods=min_periods_fwd, closed=closed, forward=True)
    fwd_start, fwd_end = roller.window.get_window_bounds()

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
        except numba.core.errors.TypingError:
            try_to_jit = False
            logging.warning('Could not jit passed statistic - omitting jitting!')

    if try_to_jit:
        stat_arr, thresh_arr = _slidingWindowSearchNumba(data_arr, bwd_start, fwd_end, split, stat_func, thresh_func,
                                                    check_len)
    else:
        stat_arr, thresh_arr = _slidingWindowSearch(data_arr, bwd_start, fwd_end, split, stat_func, thresh_func,
                                                    check_len)
    result_arr = stat_arr > thresh_arr

    if model_by_resids:
        residues = pd.Series(np.nan, index=data[field].index)
        residues[masked_index] = stat_arr
        data[field] = residues
        flagger = flagger.setFlags(field, flag=flagger.UNFLAGGED, force=True, **kwargs)
        return data, flagger

    det_index = masked_index[result_arr]
    detected = pd.Series(True, index=det_index)
    if reduce_window is not False:
        l = detected.shape[0]
        roller = customRoller(detected, window=reduce_window, min_periods=1, closed='both', center=True)
        start, end = roller.window.get_window_bounds(num_values=l, min_periods=1, closed='both', center=True)

        detected = _reduceCPCluster(stat_arr[result_arr], thresh_arr[result_arr], start, end, reduce_func, l)
        det_index = det_index[detected]

    cluster = pd.Series(False, index=data[field].index)
    cluster[det_index] = True
    cluster = cluster.cumsum()
    # (better to start cluster labels with number one)
    cluster += 1
    data[field] = cluster
    flagger = flagger.setFlags(field, flag=flagger.UNFLAGGED, force=True, **kwargs)
    if flag_changepoints:
        flagger = flagger.setFlags(field, loc=det_index)
    return data, flagger
