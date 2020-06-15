#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.core.register import register
from saqc.lib.ts_operators import polyRoller, polyRollerNoMissing, polyRoller_numba, polyRollerNoMissing_numba, \
    polyRollerIrregular, validationAgg


@register
def modelling_polyFit(data, field, flagger, winsz, polydeg, numba='auto', eval_flags=True, min_periods=0, **kwargs):
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
    data = data.copy()
    to_fit = data[field]
    flags = flagger.getFlags(field)
    if not to_fit.index.freqstr:
        if isinstance(winsz, int):
            raise NotImplementedError('Integer based window size is not supported for not-harmonized'
                             'sample series.')
        # get interval centers
        centers = np.floor((to_fit.rolling(pd.Timedelta(winsz)/2, closed='both', min_periods=min_periods).count()))
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        residues = to_fit.rolling(pd.Timedelta(winsz), closed='both', min_periods=min_periods).apply(polyRollerIrregular,
                                                                                   args=(centers, polydeg))
        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = centers.rolling('1h', closed='both').apply(center_func, raw=False).astype(int)
        temp = residues.copy()
        for k in centers_iloc.iteritems():
            residues.iloc[k[1]] = temp[k[0]]
        residues[residues.index[0]:residues.index[centers_iloc[0]]] = np.nan
        residues[residues.index[centers_iloc[-1]]:residues.index[-1]] = np.nan
    else:
        if isinstance(winsz, str):
            winsz = int(np.floor(pd.Timedelta(winsz) / pd.Timedelta(to_fit.index.freqstr)))
        if winsz % 2 == 0:
            winsz = int(winsz -1)
        if numba == 'auto':
            if to_fit.shape[0] < 200000:
                numba = False
            else:
                numba = True

        val_range = np.arange(0, winsz)
        center_index = int(np.floor(winsz / 2))
        if min_periods < winsz:
            if min_periods > 0:
                to_fit = to_fit.rolling(winsz, min_periods=min_periods, center=True).apply(lambda x, y: x[y],
                                                                                           raw=True,
                                                                                           args=(center_index,))

            # we need a missing value marker that is not nan, because nan values dont get passed by pandas rolling
            # method
            miss_marker = to_fit.min()
            miss_marker = np.floor(miss_marker - 1)
            na_mask = to_fit.isna()
            to_fit[na_mask] = miss_marker
            if numba:
                residues = to_fit.rolling(winsz).apply(polyRoller_numba, args=(miss_marker, val_range,
                                                                                            center_index, polydeg),
                                                        raw=True, engine='numba', engine_kwargs={'no_python': True})
                # due to a tiny bug - rolling with center=True doesnt work when using numba engine.
                residues = residues.shift(-int(center_index))
            else:
                residues = to_fit.rolling(winsz, center=True).apply(polyRoller, args=(miss_marker, val_range,
                                                                                      center_index, polydeg),
                                                                                             raw=True)
            residues[na_mask] = np.nan
        else:
            # we only fit fully populated intervals:
            if numba:
                residues = to_fit.rolling(winsz).apply(polyRollerNoMissing_numba, args=(val_range,
                                                                                                     center_index,
                                                                                                     polydeg),
                                                                    engine='numba', engine_kwargs={'no_python': True},
                                                                    raw=True)
                # due to a tiny bug - rolling with center=True doesnt work when using numba engine.
                residues = residues.shift(-int(center_index))
            else:
                residues = to_fit.rolling(winsz, center=True).apply(polyRollerNoMissing,
                                                                    args=(val_range, center_index, polydeg), raw=True)

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
        flagger = flagger.setFlags(field, flags=to_flag, **kwargs)

    return data, flagger
