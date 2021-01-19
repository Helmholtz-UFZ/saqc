#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from saqc.core.register import register
from saqc.lib.tools import toSequence, evalFreqStr, dropper
from saqc.lib.ts_operators import interpolateNANs


@register(masking='field')
def interpolateByRolling(
    data, field, flagger, winsz, func=np.median, center=True, min_periods=0, interpol_flag="UNFLAGGED", **kwargs
):
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
    flagger : saqc.flagger.BaseFlagger
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
    interpol_flag : {'GOOD', 'BAD', 'UNFLAGGED', str}, default 'UNFLAGGED'
        Flag that is to be inserted for the interpolated values. You can either pass one of the three major flag-classes
        or specify directly a certain flag from the passed flagger.

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

    if interpol_flag:
        if interpol_flag in ["BAD", "UNFLAGGED", "GOOD"]:
            interpol_flag = getattr(flagger, interpol_flag)
        flagger = flagger.setFlags(field, loc=interpolated, force=True, flag=interpol_flag, **kwargs)

    return data, flagger


@register(masking='field')
def interpolateInvalid(
    data,
    field,
    flagger,
    method,
    inter_order=2,
    inter_limit=2,
    interpol_flag="UNFLAGGED",
    downgrade_interpolation=False,
    not_interpol_flags=None,
    **kwargs
):

    """
    Function to interpolate nan values in the data.

    There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
    the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.

    Note, that the `inter_limit` keyword really restricts the interpolation to chunks, not containing more than
    `inter_limit` successive nan entries.

    Note, that the function differs from ``proc_interpolateGrid``, in its behaviour to ONLY interpolate nan values that
    were already present in the data passed.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated.
    interpol_flag : {'GOOD', 'BAD', 'UNFLAGGED', str}, default 'UNFLAGGED'
        Flag that is to be inserted for the interpolated values. You can either pass one of the three major flag-classes
        or specify directly a certain flag from the passed flagger.
    downgrade_interpolation : bool, default False
        If interpolation can not be performed at `inter_order` - (not enough values or not implemented at this order) -
        automaticalyy try to interpolate at order `inter_order` :math:`- 1`.
    not_interpol_flags : {None, str, List[str]}, default None
        A list of flags or a single Flag, marking values, you want NOT to be interpolated.

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
    inter_data = interpolateNANs(
        data[field],
        method,
        order=inter_order,
        inter_limit=inter_limit,
        downgrade_interpolation=downgrade_interpolation,
        return_chunk_bounds=False,
    )
    interpolated = data[field].isna() & inter_data.notna()

    if not_interpol_flags:
        for f in toSequence(not_interpol_flags):
            if f in ["BAD", "UNFLAGGED", "GOOD"]:
                f = getattr(flagger, interpol_flag)
            is_flagged = flagger.isFlagged(flag=f)[field]
            cond = is_flagged & interpolated
            inter_data.mask(cond, np.nan, inplace=True)
        interpolated &= inter_data.notna()

    if interpol_flag:
        if interpol_flag in ["BAD", "UNFLAGGED", "GOOD"]:
            interpol_flag = getattr(flagger, interpol_flag)
        flagger = flagger.setFlags(field, loc=interpolated, force=True, flag=interpol_flag, **kwargs)

    data[field] = inter_data
    return data, flagger


@register(masking='field')
def interpolateIndex(
        data,
        field,
        flagger,
        freq,
        method,
        inter_order=2,
        to_drop=None,
        downgrade_interpolation=False,
        empty_intervals_flag=None,
        grid_field=None,
        inter_limit=2,
        freq_check=None,
        **kwargs):

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

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    flagger : saqc.flagger.BaseFlagger
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
    to_drop : {None, str, List[str]}, default None
        Flags that refer to values you want to drop before interpolation - effectively excluding grid points from
        interpolation, that are only surrounded by values having a flag in them, that is listed in drop flags. Default
        results in the flaggers *BAD* flag to be the drop_flag.
    downgrade_interpolation : bool, default False
        If interpolation can not be performed at `inter_order` - (not enough values or not implemented at this order) -
        automatically try to interpolate at order `inter_order` :math:`- 1`.
    empty_intervals_flag : str, default None
        A Flag, that you want to assign to those values in the resulting equidistant sample grid, that were not
        surrounded by valid data in the original dataset, and thus were not interpolated. Default automatically assigns
        ``flagger.BAD`` flag to those values.
    grid_field : String, default None
        Use the timestamp of another variable as (not necessarily regular) "grid" to be interpolated.
    inter_limit : Integer, default 2
        Maximum number of consecutive Grid values allowed for interpolation. If set
        to *n*, chunks of *n* and more consecutive grid values, where there is no value in between, wont be
        interpolated.
    freq_check : {None, 'check', 'auto'}, default None

        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
          if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values and shape may have changed relatively to the data input.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values and shape may have changed relatively to the flagger input.
    """

    datcol = data[field]
    datcol = datcol.copy()
    flagscol = flagger.getFlags(field)
    freq = evalFreqStr(freq, freq_check, datcol.index)
    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD

    drop_mask = dropper(field, to_drop, flagger, flagger.BAD)
    drop_mask |= flagscol.isna()
    drop_mask |= datcol.isna()
    datcol[drop_mask] = np.nan
    datcol.dropna(inplace=True)
    freq = evalFreqStr(freq, freq_check, datcol.index)
    if datcol.empty:
        data[field] = datcol
        reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, inplace=True, **kwargs)
        flagger = flagger.slice(drop=field).merge(reshaped_flagger, subset=[field], inplace=True)
        return data, flagger
    # account for annoying case of subsequent frequency aligned values, differing exactly by the margin
    # 2*freq:
    spec_case_mask = datcol.index.to_series()
    spec_case_mask = spec_case_mask - spec_case_mask.shift(1)
    spec_case_mask = spec_case_mask == 2 * pd.Timedelta(freq)
    spec_case_mask = spec_case_mask[spec_case_mask]
    spec_case_mask = spec_case_mask.resample(freq).asfreq().dropna()

    if not spec_case_mask.empty:
        spec_case_mask = spec_case_mask.tshift(-1, freq)

    # prepare grid interpolation:
    if grid_field is None:
        grid_index = pd.date_range(start=datcol.index[0].floor(freq), end=datcol.index[-1].ceil(freq), freq=freq,
                                   name=datcol.index.name)
    else:
        grid_index = data[grid_field].index


    aligned_start = datcol.index[0] == grid_index[0]
    aligned_end = datcol.index[-1] == grid_index[-1]
    datcol = datcol.reindex(datcol.index.join(grid_index, how="outer",))

    # do the interpolation
    inter_data, chunk_bounds = interpolateNANs(
        datcol, method, order=inter_order, inter_limit=inter_limit, downgrade_interpolation=downgrade_interpolation,
        return_chunk_bounds=True
    )

    if grid_field is None:
        # override falsely interpolated values:
        inter_data[spec_case_mask.index] = np.nan

    # store interpolated grid
    inter_data = inter_data[grid_index]
    data[field] = inter_data

    # flags reshaping (dropping data drops):
    flagscol.drop(flagscol[drop_mask].index, inplace=True)

    if grid_field is not None:
        # only basic flag propagation supported for custom grids (take worst from preceeding/succeeding)
        preceeding = flagscol.reindex(grid_index, method='ffill', tolerance=freq)
        succeeding = flagscol.reindex(grid_index, method='bfill', tolerance=freq)
        # check for too big gaps in the source data and drop the values interpolated in those too big gaps
        na_mask = preceeding.isna() | succeeding.isna()
        na_mask = na_mask[na_mask]
        preceeding.drop(na_mask.index, inplace=True)
        succeeding.drop(na_mask.index, inplace=True)
        inter_data.drop(na_mask.index, inplace=True)
        data[field] = inter_data
        mask = succeeding > preceeding
        preceeding.loc[mask] = succeeding.loc[mask]
        flagscol = preceeding
        flagger_new = flagger.initFlags(inter_data).setFlags(field, flag=flagscol, force=True, **kwargs)
        flagger = flagger.slice(drop=field).merge(flagger_new)
        return data, flagger

    # for freq defined grids, max-aggregate flags of every grid points freq-ranged surrounding
    # hack ahead! Resampling with overlapping intervals:
    # 1. -> no rolling over categories allowed in pandas, so we translate manually:
    cats = pd.CategoricalIndex(flagger.dtype.categories, ordered=True)
    cats_dict = {cats[i]: i for i in range(0, len(cats))}
    flagscol = flagscol.replace(cats_dict)
    # 3. -> combine resample+rolling to resample with overlapping intervals:
    flagscol = flagscol.resample(freq).max()
    initial = flagscol[0]
    flagscol = flagscol.rolling(2, center=True, closed="neither").max()
    flagscol[0] = initial
    cats_dict = {num: key for (key, num) in cats_dict.items()}
    flagscol = flagscol.astype(int, errors="ignore").replace(cats_dict)
    flagscol[flagscol.isna()] = empty_intervals_flag
    # ...hack done

    # we might miss the flag for interpolated data grids last entry (if we miss it - the datapoint is always nan
    # - just settling a convention here(resulting GRID should start BEFORE first valid data entry and range to AFTER
    # last valid data)):
    if inter_data.shape[0] > flagscol.shape[0]:
        flagscol = flagscol.append(pd.Series(empty_intervals_flag, index=[datcol.index[-1]]))

    # Additional consistency operation: we have to block first/last interpolated datas flags - since they very
    # likely represent chunk starts/ends (except data start and or end timestamp were grid-aligned before Grid
    # interpolation already.)
    if np.isnan(inter_data[0]) and not aligned_start:
        chunk_bounds = chunk_bounds.insert(0, inter_data.index[0])
    if np.isnan(inter_data[-1]) and not aligned_end:
        chunk_bounds = chunk_bounds.append(pd.DatetimeIndex([inter_data.index[-1]]))
    chunk_bounds = chunk_bounds.unique()
    flagger_new = flagger.initFlags(inter_data).setFlags(field, flag=flagscol, force=True, inplace=True, **kwargs)

    # block chunk ends of interpolation
    flags_to_block = pd.Series(np.nan, index=chunk_bounds).astype(flagger_new.dtype)
    flagger_new = flagger_new.setFlags(field, loc=chunk_bounds, flag=flags_to_block, force=True, inplace=True)

    flagger = flagger.slice(drop=field).merge(flagger_new, subset=[field], inplace=True)
    return data, flagger