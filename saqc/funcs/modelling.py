#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.core.register import register
from saqc.lib.ts_operators import (
    polyRoller,
    polyRollerNoMissing,
    polyRollerNumba,
    polyRollerNoMissingNumba,
    polyRollerIrregular,
)
from saqc.lib.tools import seasonalMask


@register
def modelling_polyFit(data, field, flagger, winsz, polydeg, numba="auto", eval_flags=True, min_periods=0, **kwargs):
    """
    Function fits a polynomial model to the data and returns the residues. (field gets overridden).
    The residue for value x is calculated by fitting a polynomial of degree "polydeg" to a data slice
    of size "winsz", wich has x at its center.

    Note, that if data[field] is not alligned to an equidistant frequency grid, the window size passed,
    has to be an offset string. Also numba boost options dont apply for irregularly sampled
    timeseries.

    Note, that calculating the residues tends to be quite cost intensive - because a function fitting is perfomed for every
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
    winsz : integer or offset String
        The size of the window you want to use for fitting. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension. The window will be centered around the vaule-to-be-fitted.
        For regularly sampled timeseries the period number will be casted down to an odd number if
        even.
    polydeg : integer
        The degree of the polynomial used for fitting
    numba : {True, False, "auto"}, default "auto"
        Wheather or not to apply numbas just-in-time compilation onto the poly fit function. This will noticably
        increase the speed of calculation, if the sample size is sufficiently high.
        If "auto" is selected, numba compatible fit functions get applied for data consisiting of > 200000 samples.
    eval_flags : boolean, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
    min_periods : integer or np.nan, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.
    kwargs

    Returns
    -------

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


@register
def modelling_rollingMean(data, field, flagger, winsz, eval_flags=True, min_periods=0, center=True, **kwargs):
    """
    Models the timeseries passed with the rolling mean.

    Parameters
    ----------
    winsz : integer or offset String
        The size of the window you want to roll with. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension.
        For regularly sampled timeseries, the period number will be casted down to an odd number if
        center = True.
    eval_flags : boolean, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
        Currently not implemented in combination with not-harmonized timeseries.
    min_periods : integer, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the mean
        fitting to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present.
    center : boolean, default True
        Wheather or not to center the window the mean is calculated of around the reference value. If False,
        the reference value is placed to the right of the window (classic rolling mean with lag.)
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
    flagger : saqc.flagger
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
    flagger : saqc.flagger
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
    datcol = data[field]
    if mode == 'seasonal':
        to_mask = seasonalMask(datcol.index, season_start, season_end, include_bounds)

    elif mode == 'mask_var':
        to_mask = data[mask_var]
        to_mask_i = to_mask.index.join(datcol.index, how='inner')
        to_mask = to_mask[to_mask_i]
    else:
        raise ValueError("Keyword passed as masking mode is unknown ({})!".format(mode))

    datcol[to_mask] = np.nan
    flags_to_block = pd.Series(np.nan, index=datcol.index[to_mask]).astype(flagger.dtype)
    data[field] = datcol
    flagger = flagger.setFlags(field, loc=datcol.index[to_mask], flag=flags_to_block, force=True)

    return data, flagger