#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import floor
from typing import Tuple, Union
from typing_extensions import Literal

import numpy as np
import pandas as pd


from dios import DictOfSeries

from saqc.core.register import register

from saqc.lib.tools import getFreqDelta
from saqc.flagger import Flagger
from saqc.lib.ts_operators import polyRollerIrregular, polyRollerNumba, polyRoller, polyRollerNoMissingNumba, polyRollerNoMissing



@register(masking='field', module="curvefit")
def fitPolynomial(data: DictOfSeries, field: str, flagger: Flagger,
                  winsz: Union[int, str],
                  polydeg: int,
                  numba: Literal[True, False, "auto"]="auto",
                  eval_flags: bool=True,
                  min_periods: int=0,
                  return_residues: bool=False,
                  **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    Function fits a polynomial model to the data and returns the fitted data curve.

    The fit is calculated by fitting a polynomial of degree `polydeg` to a data slice
    of size `winsz`, that has x at its center.

    Note, that the resulting fit is stored to the `field` field of the input data, so that the original data, the
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
    flagger : saqc.flagger.Flagger
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
    min_periods : {int, None}, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.
    return_residues : bool, default False
        Internal parameter. Makes the method return the residues instead of the fit.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    """
    if data[field].empty:
        return data, flagger
    data = data.copy()
    to_fit = data[field]
    flags = flagger.getFlags(field)
    regular = getFreqDelta(to_fit.index)
    if not regular:
        if isinstance(winsz, int):
            raise NotImplementedError("Integer based window size is not supported for not-harmonized" "sample series.")
        # get interval centers
        centers = (to_fit.rolling(pd.Timedelta(winsz) / 2, closed="both", min_periods=min_periods).count()).floor()
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
            winsz = pd.Timedelta(winsz) // regular
        if winsz % 2 == 0:
            winsz = int(winsz - 1)
        if min_periods is None:
            min_periods = winsz
        if numba == "auto":
            if to_fit.shape[0] < 200000:
                numba = False
            else:
                numba = True

        val_range = np.arange(0, winsz)
        center_index = winsz // 2
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

    if return_residues:
        residues = to_fit - residues

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
