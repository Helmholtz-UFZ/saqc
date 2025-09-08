#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from saqc import UNFLAGGED
from saqc.core import register
from saqc.core.history import History
from saqc.lib.tools import isflagged
from saqc.lib.ts_operators import interpolateNANs
from saqc.lib.types import (
    FreqStr,
    Int,
    OffsetStr,
    SaQC,
    SaQCFields,
    ValidatePublicMembers,
)
from saqc.parsing.environ import ENV_OPERATORS

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


class InterpolationMixin(ValidatePublicMembers):
    @register(
        mask=["field"],
        demask=["field"],
        squeeze=[],  # func handles history by itself
    )
    def interpolateByRolling(
        self: SaQC,
        field: str,
        window: OffsetStr | (Int > 0),
        func: Callable[[pd.Series], float] | str = "median",
        center: bool = True,
        min_periods: Int >= 0 = 0,
        flag: float = UNFLAGGED,
        **kwargs,
    ) -> SaQC:
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

        if isinstance(func, str):
            func = ENV_OPERATORS[func]

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
        self: SaQC,
        field: SaQCFields,
        freq: Union[FreqStr, int],
        method: str = "time",
        order: Int > 0 = 2,
        overwrite: bool = False,
        **kwargs,
    ) -> SaQC:
        """
        Convert a time series to a specified frequency, interpolating values
        according to the chosen method.

        Parameters
        ----------
        freq : str or int
            Target frequency (e.g., "1H", "15min", 60).

        method : str
            Interpolation technique to use. Supported values include:

            * ``'nshift'``: Shift grid points to the nearest time stamp
              within +/- 0.5 * ``freq``.
            * ``'bshift'``: Shift grid points to the first succeeding time stamp.
            * ``'fshift'``: Shift grid points to the last preceding time stamp.
            * ``'linear'``, ``'time'``, ``'index'``, ``'values'``: Use numerical values
              of the index. (Note: internally mapped to ``'mshift'``.)
            * ``'pad'``: Fill NaNs using existing values (same as ``'fshift'``).
            * ``'spline'``, ``'polynomial'``: Passed to
              ``scipy.interpolate.interp1d``. Requires specifying ``order``.
            * ``'nearest'``, ``'zero'``, ``'slinear'``, ``'quadratic'``,
              ``'cubic'``, ``'barycentric'``: Passed to
              ``scipy.interpolate.interp1d``.
            * ``'krogh'``, ``'pchip'``, ``'akima'``, ``'cubicspline'``:
              Wrappers around SciPy interpolation methods.
            * ``'from_derivatives'``: Uses
              ``scipy.interpolate.BPoly.from_derivatives``.

        order : int
            Order of the interpolation method. Used only by methods that support it
            (e.g., polynomial, spline). Ignored otherwise.

        overwrite : bool
            If ``True``, existing flags will be cleared.
        """
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
    saqc: SaQC,
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
    saqc: SaQC,
    field: str,
    freq: str,
    method: str,
    order: int | None,
    dfilter: float,
    extrapolate: Literal["forward", "backward", "both", None] | None = None,
) -> Tuple[pd.Series, History]:
    """TODO: Docstring"""

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
