#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc import UNFLAGGED
from saqc.core import register
from saqc.core.history import History
from saqc.lib.checking import (
    isValidChoice,
    validateCallable,
    validateChoice,
    validateFuncSelection,
    validateMinPeriods,
    validateValueBounds,
    validateWindow,
)
from saqc.lib.tools import isflagged
from saqc.lib.ts_operators import interpolateNANs
from saqc.parsing.environ import ENV_OPERATORS

if TYPE_CHECKING:
    from saqc import SaQC


DATA_REINDEXER = {"fshift": "last", "bshift": "first", "nshift": "first"}


def _resampleOverlapping(data: pd.Series, freq: str, fill_value):
    """TODO: docstring needed"""
    dtype = data.dtype
    end = data.index[-1].ceil(freq)
    data = data.resample(freq).max()
    data = data.combine(data.shift(1, fill_value=fill_value), max)
    if end not in data:
        data.loc[end] = fill_value
    return data.fillna(fill_value).astype(dtype)


def _shift2Freq(
    data: Union[pd.Series, pd.DataFrame],
    method: Literal["fshift", "bshift", "nshift"],
    freq: str,
    fill_value,
):
    """
    shift timestamps backwards/forwards in order to align them with an equidistant
    frequency grid. Resulting Nan's are replaced with the fill-value.
    """
    validateWindow(freq, "freq", allow_int=False)
    validateChoice(method, "method", ["fshift", "bshift", "nshift"])
    methods = {
        "fshift": lambda freq: ("ffill", pd.Timedelta(freq)),
        "bshift": lambda freq: ("bfill", pd.Timedelta(freq)),
        "nshift": lambda freq: ("nearest", pd.Timedelta(freq) / 2),
    }
    direction, tolerance = methods[method](freq)
    target_ind = pd.date_range(
        start=pd.Timestamp(data.index[0]).floor(freq),
        end=pd.Timestamp(data.index[-1]).ceil(freq),
        freq=freq,
        name=data.index.name,
    )
    return data.reindex(
        target_ind, method=direction, tolerance=tolerance, fill_value=fill_value
    )


class InterpolationMixin:
    @register(
        mask=["field"],
        demask=["field"],
        squeeze=[],  # func handles history by itself
    )
    def interpolateByRolling(
        self: "SaQC",
        field: str,
        window: str | int,
        func: Callable[[pd.Series], float] | str = "median",
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> "SaQC":
        """
        Replace NaN by the aggregation result of the surrounding window.

        Parameters
        ----------
        window :
            The size of the window, the aggregation is computed from.
            An integer define the number of periods to be used, a string
            is interpreted as an offset. ( see `pandas.rolling` for more
            information). Integer windows may result in screwed aggregations
            if called on none-harmonized or irregular data.

        func : default median
            The function used for aggregation.

        center :
            Center the window around the value. Can only be used with
            integer windows, otherwise it is silently ignored.

        min_periods :
            Minimum number of valid (not np.nan) values that have to be
            available in a window for its aggregation to be
            computed.
        """
        validateWindow(window)
        validateFuncSelection(func, allow_operator_str=True)
        if isinstance(func, str):
            func = ENV_OPERATORS[func]
        validateMinPeriods(min_periods)

        datcol = self._data[field]
        roller = datcol.rolling(window=window, center=center, min_periods=min_periods)
        try:
            func_name = func.__name__
            if func_name[:3] == "nan":
                func_name = func_name[3:]
            rolled = getattr(roller, func_name)()
        except AttributeError:
            rolled = roller.apply(func)

        na_mask = datcol.isna()
        interpolated = na_mask & rolled.notna()
        datcol[na_mask] = rolled[na_mask]
        self._data[field] = datcol

        flagcol = pd.Series(np.nan, index=self._flags[field].index)
        flagcol.loc[interpolated] = np.nan if flag is None else flag

        meta = {
            "func": "interpolateByRolling",
            "args": (field,),
            "kwargs": {
                "window": window,
                "func": func,
                "center": center,
                "min_periods": min_periods,
                "flag": flag,
                **kwargs,
            },
        }
        self._flags.history[field].append(flagcol, meta)

        return self

    @register(mask=["field"], demask=[], squeeze=[])
    def align(
        self: "SaQC",
        field: str,
        freq: str,
        method: str = "time",
        order: int = 2,
        overwrite: bool = False,
        **kwargs,
    ) -> "SaQC":
        """
        Convert time series to specified frequency. Values affected by
        frequency changes will be inteprolated using the given method.

        Parameters
        ----------
        freq :
            Target frequency.

        method :
            Interpolation technique to use. One of:

            * ``'nshift'``: Shift grid points to the nearest time stamp
              in the range = +/- 0.5 * ``freq``.
            * ``'bshift'``: Shift grid points to the first succeeding
              time stamp (if any).
            * ``'fshift'``: Shift grid points to the last preceeding time
              stamp (if any).
            * ``'linear'``: Ignore the index and treat the values as equally
              spaced.
            * ``'time'``, ``'index'``, ``'values'``: Use the actual numerical
              values of the index.
            * ``'pad'``: Fill in NaNs using existing values.
            * ``'spline'``, ``'polynomial'``:
              Passed to ``scipy.interpolate.interp1d``. These methods
              use the numerical values of the index.  An ``order`` must be
              specified, e.g. ``qc.interpolate(method='polynomial', order=5)``.
            * ``'nearest'``, ``'zero'``, ``'slinear'``, ``'quadratic'``, ``'cubic'``, ``'barycentric'``:
              Passed to ``scipy.interpolate.interp1d``. These methods use
              the numerical values of the index.
            * ``'krogh'``, ``'spline'``, ``'pchip'``, ``'akima'``, ``'cubicspline'``:
              Wrappers around the SciPy interpolation methods of similar
              names.
            * ``'from_derivatives'``: Refers to ``scipy.interpolate.BPoly.from_derivatives``.

        order :
            Order of the interpolation method, ignored if not supported
            by the chosen ``method``.

        extrapolate :
            Use parameter to perform extrapolation instead of interpolation
            onto the trailing and/or leading chunks of NaN values in data series.

            * ``None`` (default) - perform interpolation
            * ``'forward'``/``'backward'`` - perform forward/backward extrapolation
            * ``'both'`` - perform forward and backward extrapolation

        overwrite :
           If set to `True`, existing flags will be cleared.
        """

        validateWindow(freq, "freq", allow_int=False)
        validateValueBounds(order, "order", left=0, strict_int=True)

        method = "fshift" if method == "pad" else method
        if method in ["time", "linear"]:
            data_agg_func = method
            method = "mshift"
        else:
            data_agg_func = DATA_REINDEXER.get(method, None)

        self = self.reindex(
            field,
            index=freq,
            tolerance=freq,
            method=method,
            override=overwrite,
            data_aggregation=data_agg_func,
            **kwargs,
        )
        return self


def _shift(
    saqc: "SaQC",
    field: str,
    freq: str,
    method: Literal["fshift", "bshift", "nshift"] = "nshift",
    **kwargs,
) -> Tuple[pd.Series, History]:
    """
    Shift data points and flags to a regular frequency grid.

    Parameters
    ----------
    field :
        The fieldname of the column, holding the data-to-be-shifted.

    freq :
        Offset string. Sampling rate of the target frequency.

    method :
        Method to propagate values:

        * 'nshift' : shift grid points to the nearest time stamp in the range = +/- 0.5 * ``freq``
        * 'bshift' : shift grid points to the first succeeding time stamp (if any)
        * 'fshift' : shift grid points to the last preceding time stamp (if any)

    Returns
    -------
    saqc.SaQC
    """
    validateChoice(method, "method", ["fshift", "bshift", "nshift"])
    validateWindow(freq, "freq", allow_int=False)

    datcol = saqc._data[field]
    if datcol.empty:
        return saqc

    # do the shift
    datcol = _shift2Freq(datcol, method, freq, fill_value=np.nan)

    # do the shift on the history
    kws = dict(method=method, freq=freq)

    history = saqc._flags.history[field].apply(
        index=datcol.index,
        func_handle_df=True,
        func=_shift2Freq,
        func_kws={**kws, "fill_value": np.nan},
    )

    return datcol, history


def _interpolate(
    saqc: "SaQC",
    field: str,
    freq: str,
    method: str,
    order: int | None,
    dfilter: float,
    extrapolate: Literal["forward", "backward", "both", None] = None,
) -> Tuple[pd.Series, History]:
    """TODO: Docstring"""

    validateChoice(extrapolate, "extrapolate", ["forward", "backward", "both", None])
    validateWindow(freq, "freq", allow_int=False)
    if order is not None:
        validateValueBounds(order, "order", 0, strict_int=True)

    datcol = saqc._data[field].copy()

    start, end = datcol.index[0].floor(freq), datcol.index[-1].ceil(freq)
    grid_index = pd.date_range(start=start, end=end, freq=freq, name=datcol.index.name)

    flagged = isflagged(saqc._flags[field], dfilter)

    # drop all points that hold no relevant grid information
    datcol = datcol[~flagged].dropna()

    # account for annoying case of subsequent frequency aligned values,
    # that differ exactly by the margin of 2*freq
    gaps = datcol.index[1:] - datcol.index[:-1] == 2 * pd.Timedelta(freq)
    gaps = datcol.index[1:][gaps]
    gaps = gaps.intersection(grid_index).shift(-1, freq)

    # prepare grid interpolation:
    datcol = datcol.reindex(datcol.index.union(grid_index))

    # do the grid interpolation
    inter_data = interpolateNANs(
        data=datcol,
        method=method,
        order=order,
        gap_limit=2,
        extrapolate=extrapolate,
    )
    # override falsely interpolatet values:
    inter_data[gaps] = np.nan
    inter_data = inter_data[grid_index]

    history = saqc._flags.history[field].apply(
        index=inter_data.index,
        func=_resampleOverlapping,
        func_kws=dict(freq=freq, fill_value=np.nan),
    )
    return inter_data, history
