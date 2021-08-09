#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable, Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import flagging, processing, Flags
from saqc.core.register import _isflagged
from saqc.lib.ts_operators import interpolateNANs

_SUPPORTED_METHODS = Literal[
    "linear",
    "time",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
]


@flagging(masking="field", module="interpolation")
def interpolateByRolling(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    winsz: Union[str, int],
    func: Callable[[pd.Series], float] = np.median,
    center: bool = True,
    min_periods: int = 0,
    flag: float = UNFLAGGED,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Interpolates nan-values in the data by assigning them the aggregation result of the window surrounding them.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        Name of the column, holding the data-to-be-interpolated.

    flags : saqc.Flags
        A flags object, holding flags and additional Information related to `data`.

    winsz : int, str
        The size of the window, the aggregation is computed from. An integer define the number of periods to be used,
        an string is interpreted as an offset. ( see `pandas.rolling` for more information).
        Integer windows may result in screwed aggregations if called on none-harmonized or irregular data.

    func : Callable
        The function used for aggregation.

    center : bool, default True
        Center the window around the value. Can only be used with integer windows, otherwise it is silently ignored.

    min_periods : int
        Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
        computed.

    flag : float or None, default UNFLAGGED
        Flag that is to be inserted for the interpolated values. If ``None`` no flags are set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """

    data = data.copy()
    datcol = data[field]
    roller = datcol.rolling(window=winsz, center=center, min_periods=min_periods)
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
    data[field] = datcol

    if flag is not None:
        flags[interpolated, field] = flag

    return data, flags


@flagging(masking="field", module="interpolation")
def interpolateInvalid(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    method: _SUPPORTED_METHODS,
    inter_order: int = 2,
    inter_limit: int = 2,
    downgrade_interpolation: bool = False,
    flag: float = UNFLAGGED,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Function to interpolate nan values in the data.

    There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
    the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        Name of the column, holding the data-to-be-interpolated.

    flags : saqc.Flags
        A flags object, holding flags and additional Information related to `data`.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
        The interpolation method to use.

    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `inter_limit` successive nan entries.

    flag : float or None, default UNFLAGGED
        Flag that is set for interpolated values. If ``None``, no flags are set at all.

    downgrade_interpolation : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``inter_order``, or
        simply because not enough values are present in a interval.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """
    inter_data = interpolateNANs(
        data[field],
        method,
        order=inter_order,
        inter_limit=inter_limit,
        downgrade_interpolation=downgrade_interpolation,
    )
    interpolated = data[field].isna() & inter_data.notna()

    if flag is not None:
        flags[interpolated, field] = flag

    data[field] = inter_data
    return data, flags


def _resampleOverlapping(data: pd.Series, freq: str, fill_value):
    """TODO: docstring needed"""
    dtype = data.dtype
    end = data.index[-1].ceil(freq)
    data = data.resample(freq).max()
    data = data.combine(data.shift(1, fill_value=fill_value), max)
    if end not in data:
        data.loc[end] = fill_value
    return data.fillna(fill_value).astype(dtype)


@processing(module="interpolation")
def interpolateIndex(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    freq: str,
    method: _SUPPORTED_METHODS,
    inter_order: int = 2,
    inter_limit: int = 2,
    downgrade_interpolation: bool = False,
    **kwargs
) -> Tuple[DictOfSeries, Flags]:
    """
    Function to interpolate the data at regular (equidistant) timestamps (or Grid points).

    Note, that the interpolation will only be calculated, for grid timestamps that have a preceding AND a succeeding
    valid data value within "freq" range.

    Parameters
    ----------
    data : dios.DictOfSeries
        The data container.

    field : str
        Name of the column, holding the data-to-be-interpolated.

    flags : saqc.Flags
        A flags object, holding flags and additional Information related to `data`.

    freq : str
        An Offset String, interpreted as the frequency of
        the grid you want to interpolate your data at.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.

    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `inter_limit` successive nan entries.

    downgrade_interpolation : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``inter_order``, or
        simply because not enough values are present in a interval.


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        Flags values and shape may have changed relatively to the flags input.
    """
    if data[field].empty:
        return data, flags

    datcol = data[field].copy()

    start, end = datcol.index[0].floor(freq), datcol.index[-1].ceil(freq)
    grid_index = pd.date_range(start=start, end=end, freq=freq, name=datcol.index.name)

    flagged = _isflagged(flags[field], kwargs["to_mask"])

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
        order=inter_order,
        inter_limit=inter_limit,
        downgrade_interpolation=downgrade_interpolation,
    )

    # override falsely interpolated values:
    inter_data[gaps] = np.nan

    # store interpolated grid
    data[field] = inter_data[grid_index]

    history = flags.history[field].apply(
        index=data[field].index,
        hist_func=_resampleOverlapping,
        mask_func=_resampleOverlapping,
        hist_kws=dict(freq=freq, fill_value=UNFLAGGED),
        mask_kws=dict(freq=freq, fill_value=True),
        copy=False,
    )

    flags.history[field] = history
    return data, flags
