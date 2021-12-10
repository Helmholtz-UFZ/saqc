#! /usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable
from typing_extensions import Literal
import numpy as np
import pandas as pd
from dios import DictOfSeries

from saqc.constants import *
from saqc.core import register, Flags
from saqc.core.register import _isflagged, processing
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


@register(
    mask=["field"],
    demask=["field"],
    squeeze=[],  # func handles history by itself
)
def interpolateByRolling(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[str, int],
    func: Callable[[pd.Series], float] = np.median,
    center: bool = True,
    min_periods: int = 0,
    flag: float = UNFLAGGED,
    **kwargs,
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

    window : int, str
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
        Flag that is to be inserted for the interpolated values.
        If `None` the old flags are kept, even if the data is valid now.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
    """
    datcol = data[field]
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
    data[field] = datcol

    new_col = pd.Series(np.nan, index=flags[field].index)
    new_col.loc[interpolated] = np.nan if flag is None else flag

    # todo kwargs must have all passed args except data,field,flags
    flags.history[field].append(
        new_col, {"func": "interpolateByRolling", "args": (), "kwargs": kwargs}
    )

    return data, flags


@register(
    mask=["field"],
    demask=["field"],
    squeeze=[],  # func handles history by itself
)
def interpolateInvalid(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    method: _SUPPORTED_METHODS,
    order: int = 2,
    limit: int = 2,
    downgrade: bool = False,
    flag: float = UNFLAGGED,
    **kwargs,
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

    order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `limit` successive nan entries.

    flag : float or None, default UNFLAGGED
        Flag that is set for interpolated values. If ``None``, no flags are set at all.

    downgrade : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``order``, or
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
        order=order,
        inter_limit=limit,
        downgrade_interpolation=downgrade,
    )
    data[field] = inter_data

    interpolated = data[field].isna() & inter_data.notna()
    new_col = pd.Series(np.nan, index=flags[field].index)
    new_col.loc[interpolated] = np.nan if flag is None else flag

    # todo kwargs must have all passed args except data,field,flags
    flags.history[field].append(
        new_col, {"func": "interpolateInvalid", "args": (), "kwargs": kwargs}
    )

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


@register(mask=["field"], demask=[], squeeze=[])
def interpolateIndex(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    freq: str,
    method: _SUPPORTED_METHODS,
    order: int = 2,
    limit: int = 2,
    downgrade: bool = False,
    **kwargs,
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

    order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `limit` successive nan entries.

    downgrade : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``order``, or
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

    # todo:
    #  in future we could use `register(mask=[field], [], [])`
    #  and dont handle masking manually here
    flagged = _isflagged(flags[field], kwargs["dfilter"])

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
        inter_limit=limit,
        downgrade_interpolation=downgrade,
    )

    # override falsely interpolated values:
    inter_data[gaps] = np.nan

    # store interpolated grid
    data[field] = inter_data[grid_index]

    history = flags.history[field].apply(
        index=data[field].index,
        func=_resampleOverlapping,
        func_kws=dict(freq=freq, fill_value=UNFLAGGED),
    )

    flags.history[field] = history
    return data, flags
