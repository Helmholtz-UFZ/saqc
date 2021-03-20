#! /usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from typing import Tuple, Union, Optional, Any, Callable, Sequence
from typing_extensions import Literal

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.constants import *
from saqc.core.register import register, isflagged
from saqc.flagger import Flagger
from saqc.flagger.flags import applyFunctionOnHistory

from saqc.lib.tools import toSequence, evalFreqStr, getDropMask
from saqc.lib.ts_operators import interpolateNANs

_SUPPORTED_METHODS = Literal[
    "linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
    "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"
]


@register(masking='field', module="interpolation")
def interpolateByRolling(
        data: DictOfSeries, field: str, flagger: Flagger,
        winsz: Union[str, int],
        func: Callable[[pd.Series], float] = np.median,
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Interpolates missing values (nan values present in the data) by assigning them the aggregation result of
    a window surrounding them.

    Note, that in the current implementation, center=True can only be used with integer window sizes - furthermore
    note, that integer window sizes can yield screwed aggregation results for not-harmonized or irregular data.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    winsz : int, str
        The size of the window, the aggregation is computed from. Either counted in periods number (Integer passed),
        or defined by a total temporal extension (offset String passed).
    func : Callable
        The function used for aggregation.
    center : bool, default True
        Wheather or not the window, the aggregation is computed of, is centered around the value to be interpolated.
    min_periods : int
        Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
        computed.
    flag : float, default UNFLAGGED
        Flag that is to be inserted for the interpolated values. If ``None`` no flags are set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
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
    interpolated = na_mask & ~rolled.isna()
    datcol[na_mask] = rolled[na_mask]
    data[field] = datcol

    if flag is not None:
        flagger[interpolated, field] = flag

    return data, flagger


@register(masking='field', module="interpolation")
def interpolateInvalid(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        method: _SUPPORTED_METHODS,
        inter_order: int = 2,
        inter_limit: int = 2,
        downgrade_interpolation: bool = False,
        not_interpol_flags=None,
        flag: float = UNFLAGGED,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function to interpolate nan values in the data.

    There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
    the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.

    Note, that the `inter_limit` keyword really restricts the interpolation to chunks, not containing more than
    `inter_limit` successive nan entries.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated.
    flag : float or None, default UNFLAGGED
        Flag that is to be inserted for the interpolated values. If ``None`` no flags are set.
    downgrade_interpolation : bool, default False
        If interpolation can not be performed at `inter_order`, because not enough values are present or the order
        is not implemented for the passed method, automatically try to interpolate at ``inter_order-1``.
    not_interpol_flags : None
        deprecated

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    data = data.copy()
    inter_data = interpolateNANs(
        data[field],
        method,
        order=inter_order,
        inter_limit=inter_limit,
        downgrade_interpolation=downgrade_interpolation
    )
    interpolated = data[field].isna() & inter_data.notna()

    # TODO: remove with version 2.0
    if not_interpol_flags is not None:
        raise ValueError("'not_interpol_flags' is deprecated")

    if flag is not None:
        flagger[interpolated, field] = flag

    data[field] = inter_data
    return data, flagger


def _overlap_rs(x, freq='1min', fill_value=UNFLAGGED):
    end = x.index[-1].ceil(freq)
    x = x.resample(freq).max()
    x = x.combine(x.shift(1, fill_value=fill_value), max)
    # we are appending last regular grid entry (if necessary), to conserve integrity of groups of regularized
    # timestamps originating all from the same logger.
    try:
        x = x.append(pd.Series([fill_value], index=[end]), verify_integrity=True)
    except ValueError:
        pass
    return x


@register(masking='none', module="interpolation")
def interpolateIndex(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        freq: str,
        method: _SUPPORTED_METHODS,
        inter_order: int = 2,
        downgrade_interpolation: bool = False,
        inter_limit: int = 2,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function to interpolate the data at regular (equidistant) timestamps (or Grid points).

    Note, that the interpolation will only be calculated, for grid timestamps that have a preceding AND a succeeding
    valid data value within "freq" range.

    Note, that the function differs from proc_interpolateMissing, by returning a whole new data set, only containing
    samples at the interpolated, equidistant timestamps (of frequency "freq").

    Note, it is possible to interpolate unregular "grids" (with no frequencies). In fact, any date index
    can be target of the interpolation. Just pass the field name of the variable, holding the index
    you want to interpolate, to "grid_field". 'freq' is then use to determine the maximum gap size for
    a grid point to be interpolated.

    Note, that intervals, not having an interpolation value assigned (thus, evaluate to np.nan), get UNFLAGGED assigned.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    freq : str
        An Offset String, interpreted as the frequency of
        the grid you want to interpolate your data at.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    inter_order : integer, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    downgrade_interpolation : bool, default False
        If interpolation can not be performed at `inter_order` - (not enough values or not implemented at this order) -
        automatically try to interpolate at order `inter_order` :math:`- 1`.
    inter_limit : Integer, default 2
        Maximum number of consecutive Grid values allowed for interpolation. If set
        to *n*, chunks of *n* and more consecutive grid values, where there is no value in between, wont be
        interpolated.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """
    if data[field].empty:
        return data, flagger

    datcol = data[field].copy()
    flagscol = flagger[field]

    start, end = datcol.index[0].floor(freq), datcol.index[-1].ceil(freq)
    grid_index = pd.date_range(start=start, end=end, freq=freq, name=datcol.index.name)

    flagged = isflagged(flagscol, kwargs['to_mask'])

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

    # flags reshaping
    flagscol = flagscol[~flagged]

    flagscol = _overlap_rs(flagscol, freq, UNFLAGGED)
    flagger = applyFunctionOnHistory(
        flagger, field,
        hist_func=_overlap_rs, hist_kws=dict(freq=freq, fill_value=UNFLAGGED),
        mask_func=_overlap_rs, mask_kws=dict(freq=freq, fill_value=False),
        last_column=flagscol
    )

    return data, flagger
