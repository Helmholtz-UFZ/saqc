#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Union
from typing_extensions import Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import register, Flags
from saqc.lib.tools import getFreqDelta, filterKwargs
from saqc.lib.ts_operators import (
    polyRollerIrregular,
    polyRollerNumba,
    polyRoller,
    polyRollerNoMissingNumba,
    polyRollerNoMissing,
)


@register(mask=["field"], demask=[], squeeze=[])
def fitPolynomial(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: int | str,
    order: int,
    min_periods: int = 0,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Fits a polynomial model to the data.

    The fit is calculated by fitting a polynomial of degree `order` to a data slice
    of size `window`, that has x at its center.

    Note that the result is stored in `field` and overwrite it unless a
    `target` is given.

    In case your data is sampled at an equidistant frequency grid:

    (1) If you know your data to have no significant number of missing values,
    or if you do not want to calculate residues for windows containing missing values
    any way, performance can be increased by setting min_periods=window.

    Note, that the initial and final window/2 values do not get fitted.

    Each residual gets assigned the worst flag present in the interval of
    the original data.

    Parameters
    ----------
    data : DictOfSeries
        The data container.

    field : str
        A column in flags and data.

    flags : Flags
        The flags container.

    window : str, int
        Size of the window you want to use for fitting. If an integer is passed,
        the size refers to the number of periods for every fitting window. If an
        offset string is passed, the size refers to the total temporal extension. The
        window will be centered around the vaule-to-be-fitted. For regularly sampled
        data always a odd number of periods will be used for the fit (periods-1 if
        periods is even).

    order : int
        Degree of the polynomial used for fitting

    min_periods : int or None, default 0
        Minimum number of observations in a window required to perform the fit,
        otherwise NaNs will be assigned.
        If ``None``, `min_periods` defaults to 1 for integer windows and to the
        size of the window for offset based windows.
        Passing 0, disables the feature and will result in over-fitting for too
        sparse windows.

    Returns
    -------
    data : dios.DictOfSeries
        Modified data
    flags : saqc.Flags
        Flags
    """
    reserved = ["residues", "set_flags"]
    filterKwargs(kwargs, reserved)
    return _fitPolynomial(
        data=data,
        field=field,
        flags=flags,
        window=window,
        order=order,
        min_periods=min_periods,
        **kwargs,
        # ctrl args
        return_residues=False,
        set_flags=True,
    )


def _fitPolynomial(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[int, str],
    order: int,
    set_flags: bool = True,
    min_periods: int = 0,
    return_residues: bool = False,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:

    # TODO: some (rather large) parts are functional similar to saqc.funcs.rolling.roll
    if data[field].empty:
        return data, flags

    to_fit = data[field].copy()
    regular = getFreqDelta(to_fit.index)
    if not regular:
        if isinstance(window, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                "sample series."
            )
        # get interval centers
        centers = (
            to_fit.rolling(
                pd.Timedelta(window) / 2, closed="both", min_periods=min_periods
            ).count()
        ).floor()
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        residues = to_fit.rolling(
            pd.Timedelta(window), closed="both", min_periods=min_periods
        ).apply(polyRollerIrregular, args=(centers, order))

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = (
            centers.rolling(window, closed="both")
            .apply(center_func, raw=False)
            .astype(int)
        )
        temp = residues.copy()
        for k in centers_iloc.iteritems():
            residues.iloc[k[1]] = temp[k[0]]
        residues[residues.index[0] : residues.index[centers_iloc[0]]] = np.nan
        residues[residues.index[centers_iloc[-1]] : residues.index[-1]] = np.nan
    else:
        if isinstance(window, str):
            window = pd.Timedelta(window) // regular
        if window % 2 == 0:
            window = int(window - 1)
        if min_periods is None:
            min_periods = window
        if to_fit.shape[0] < 200000:
            numba = False
        else:
            numba = True

        val_range = np.arange(0, window)
        center_index = window // 2
        if min_periods < window:
            if min_periods > 0:
                to_fit = to_fit.rolling(
                    window, min_periods=min_periods, center=True
                ).apply(lambda x, y: x[y], raw=True, args=(center_index,))

            # we need a missing value marker that is not nan,
            # because nan values dont get passed by pandas rolling method
            miss_marker = to_fit.min()
            miss_marker = np.floor(miss_marker - 1)
            na_mask = to_fit.isna()
            to_fit[na_mask] = miss_marker
            if numba:
                residues = to_fit.rolling(window).apply(
                    polyRollerNumba,
                    args=(miss_marker, val_range, center_index, order),
                    raw=True,
                    engine="numba",
                    engine_kwargs={"no_python": True},
                )
                # due to a tiny bug - rolling with center=True doesnt work
                # when using numba engine.
                residues = residues.shift(-int(center_index))
            else:
                residues = to_fit.rolling(window, center=True).apply(
                    polyRoller,
                    args=(miss_marker, val_range, center_index, order),
                    raw=True,
                )
            residues[na_mask] = np.nan
        else:
            # we only fit fully populated intervals:
            if numba:
                residues = to_fit.rolling(window).apply(
                    polyRollerNoMissingNumba,
                    args=(val_range, center_index, order),
                    engine="numba",
                    engine_kwargs={"no_python": True},
                    raw=True,
                )
                # due to a tiny bug - rolling with center=True doesnt work
                # when using numba engine.
                residues = residues.shift(-int(center_index))
            else:
                residues = to_fit.rolling(window, center=True).apply(
                    polyRollerNoMissing,
                    args=(val_range, center_index, order),
                    raw=True,
                )

    if return_residues:
        residues = to_fit - residues

    data[field] = residues
    if set_flags:
        # TODO: we does not get any flags here, because of masking=field
        worst = flags[field].rolling(window, center=True, min_periods=min_periods).max()
        flags[field] = worst

    return data, flags
