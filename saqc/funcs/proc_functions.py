#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.core.register import register
from saqc.lib.ts_operators import interpolateNANs, aggregate2Freq, shift2Freq, expModelFunc
from saqc.lib.tools import toSequence, mergeDios, dropper, mutateIndex
import dios
import functools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
ORIGINAL_SUFFIX = '_original'

METHOD2ARGS = {'inverse_fshift': ('backward', pd.Timedelta),
               'inverse_bshift': ('forward', pd.Timedelta),
               'inverse_nshift': ('nearest', lambda x: pd.Timedelta(x)/2),
               'inverse_fagg': ('bfill', pd.Timedelta),
               'inverse_bagg': ('ffill', pd.Timedelta),
               'inverse_nagg': ('nearest', lambda x: pd.Timedelta(x)/2),
               'match': (None, lambda x: '0min')}


@register
def proc_rollingInterpolateMissing(data, field, flagger, winsz, func=np.median, center=True, min_periods=0,
                                   interpol_flag='UNFLAGGED', **kwargs):
    """
    Interpolates missing values (nan values present in the data) by assigning them the aggregation result of
    a window surrounding them.

    Note, that in the current implementation, center=True can only be used with integer window sizes - furthermore
    note, that integer window sizes can yield screwed aggregation results for not-harmonized or irregular data.



    Parameters
    ----------
    winsz : Integer or Offset String
        The size of the window, the aggregation is computed from. Either counted in periods number (Integer passed),
        or defined by a total temporal extension (offset String passed)
    func : Function
        The function used for aggregation.
    center : boolean, default True
        Wheather or not the window the aggregation is computed of is centered around the value to be interpolated.
    min_periods : Integer
        Minimum number of valid (not np.nan) values that have to be available in a window for its Aggregation to be
        computed.
    interpol_flag : {'GOOD', 'BAD', 'UNFLAGGED'} or String, default 'UNFLAGGED'
        Flag that is to be inserted for the interpolated values. You can either pass one of the three major flag-classes
        or specify directly a certain flag from the passed flagger.

    Returns
    -------

    """
    data = data.copy()
    datcol = data[field]
    roller = datcol.rolling(window=winsz, center=center, min_periods=min_periods)
    try:
        func_name = func.__name__
        if func_name[:3] == 'nan':
            func_name = func_name[3:]
        rolled = getattr(roller, func_name)()
    except AttributeError:
        rolled = roller.apply(func)

    na_mask = datcol.isna()
    interpolated = na_mask & ~rolled.isna()
    datcol[na_mask] = rolled[na_mask]
    data[field] = datcol

    if interpol_flag:
        if interpol_flag in ['BAD', 'UNFLAGGED', 'GOOD']:
            interpol_flag = getattr(flagger, interpol_flag)
        flagger = flagger.setFlags(field, loc=interpolated[interpolated].index, force=True,
                                   flag=interpol_flag, **kwargs)

    return data, flagger


@register
def proc_interpolateMissing(data, field, flagger, method, inter_order=2, inter_limit=2, interpol_flag='UNFLAGGED',
                            downgrade_interpolation=False, not_interpol_flags=None, **kwargs):

    """
    function to interpolate nan values in the data.
    There are available all the interpolation methods from the pandas.interpolate() method and they are applicable by
    the very same key words, that you would pass to pd.Series.interpolates's method parameter.

    Note, that the inter_limit keyword really restricts the interpolation to chunks, not containing more than
    "inter_limit" successive nan entries.

    Note, that the function differs from proc_interpolateGrid, in its behaviour to ONLY interpolate nan values that
    were already present in the data passed.

    Parameters
    ---------
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
    "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.

    inter_order : integer, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    inter_limit : integer, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated.

    interpol_flag : {'GOOD', 'BAD', 'UNFLAGGED'} or String, default 'UNFLAGGED'
        Flag that is to be inserted for the interpolated values. You can either pass one of the three major flag-classes
        or specify directly a certain flag from the passed flagger.

    downgrade_interpolation : boolean, default False
        If interpolation can not be performed at 'inter_order' - (not enough values or not implemented at this order) -
        automaticalyy try to interpolate at order 'inter_order' - 1.

    not_interpol_flags : list or String, default None
        A list of flags or a single Flag, marking values, you want NOT to be interpolated.
    """

    data = data.copy()
    inter_data = interpolateNANs(data[field], method, order=inter_order, inter_limit=inter_limit,
                           downgrade_interpolation=downgrade_interpolation, return_chunk_bounds=False)
    interpolated = data[field].isna() & inter_data.notna()

    if not_interpol_flags:
        for f in toSequence(not_interpol_flags):
            if f in ['BAD', 'UNFLAGGED', 'GOOD']:
                f = getattr(flagger, interpol_flag)
            is_flagged = flagger.isFlagged(flag=f)[field]
            cond = is_flagged & interpolated
            inter_data.mask(cond, np.nan, inplace=True)
        interpolated &= inter_data.notna()

    if interpol_flag:
        if interpol_flag in ['BAD', 'UNFLAGGED', 'GOOD']:
            interpol_flag = getattr(flagger, interpol_flag)
        flagger = flagger.setFlags(field, loc=interpolated[interpolated].index, force=True,
                                   flag=interpol_flag, **kwargs)

    data[field] = inter_data
    return data, flagger


@register
def proc_interpolateGrid(data, field, flagger, freq, method, inter_order=2, drop_flags=None,
                            downgrade_interpolation=False, empty_intervals_flag=None, **kwargs):
    """
    Function to interpolate the data at regular (equidistant) timestamps (or Grid points).

    Note, that the interpolation will only be calculated, for grid timestamps that have a preceeding AND a succeeding
    valid data value within "freq" range.

    Note, that the function differs from proc_interpolateMissing, by returning a whole new data set, only containing
    samples at the interpolated, equidistant timestamps (of frequency "freq").

    Parameters
    ---------
    freq : Offset String
        The frequency of the grid you want to interpolate your data at.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.

    inter_order : integer, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    drop_flags : list or string, default None
        Flags that refer to values you want to drop before interpotion - effectively excluding grid points from
        interpolation, that are only surrounded by values having a flag in them, that is listed in drop flags. Default
        results in the flaggers 'BAD' flag to be the drop_flag.

    downgrade_interpolation : boolean, default False
        If interpolation can not be performed at 'inter_order' - (not enough values or not implemented at this order) -
        automaticalyy try to interpolate at order 'inter_order' - 1.

    empty_intervals_flag : String, default None
        A Flag, that you want to assign to those values resulting equidistant sample grid, that were not surrounded by
        valid (flagged) data in the original dataset and thus werent interpolated. Default automatically assigns
        flagger.BAD flag to those values.
        """

    datcol = data[field]
    datcol = datcol.copy()
    flagscol = flagger.getFlags(field)
    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD

    drop_mask = dropper(field, drop_flags, flagger, flagger.BAD)
    drop_mask |= flagscol.isna()
    drop_mask |= datcol.isna()
    datcol[drop_mask] = np.nan
    datcol.dropna(inplace=True)
    if datcol.empty:
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
    grid_index = pd.date_range(start=datcol.index[0].floor(freq), end=datcol.index[-1].ceil(freq), freq=freq,
                               name=datcol.index.name)

    aligned_start = datcol.index[0] == grid_index[0]
    aligned_end = datcol.index[-1] == grid_index[-1]
    datcol = datcol.reindex(
        datcol.index.join(grid_index, how="outer", )
    )

    # do the interpolation
    inter_data, chunk_bounds = interpolateNANs(
        datcol, method, order=inter_order, inter_limit=2, downgrade_interpolation=downgrade_interpolation,
        return_chunk_bounds=True
    )

    # override falsely interpolated values:
    inter_data[spec_case_mask.index] = np.nan

    # store interpolated grid
    inter_data = inter_data.asfreq(freq)
    data[field] = inter_data

    # flags reshaping (dropping data drops):
    flagscol.drop(flagscol[drop_mask].index, inplace=True)

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
    flagscol = flagscol.astype(int, errors='ignore').replace(cats_dict)
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
    flagger_new = flagger.initFlags(inter_data).setFlags(field, flag=flagscol, force=True, **kwargs)

    # block chunk ends of interpolation
    flags_to_block = pd.Series(np.nan, index=chunk_bounds).astype(flagger_new.dtype)
    flagger_new = flagger_new.setFlags(field, loc=chunk_bounds, flag=flags_to_block, force=True)

    flagger = flagger.slice(drop=field).merge(flagger_new)
    return data, flagger


@register
def proc_resample(data, field, flagger, freq, agg_func=np.mean, method='bagg', max_invalid_total_d=np.inf,
                  max_invalid_consec_d=np.inf, max_invalid_consec_f=np.inf, max_invalid_total_f=np.inf,
                  flag_agg_func=max, empty_intervals_flag=None, drop_flags=None, **kwargs):
    """
    Function to resample the data. Afterwards the data will be sampled at regular (equidistant) timestamps
    (or Grid points). Sampling intervals therefor get aggregated with a function, specifyed by 'agg_func' parameter and
    the result gets projected onto the new timestamps with a method, specified by "method". The following method
    (keywords) are available:

    'nagg' - all values in the range (+/- freq/2) of a grid point get aggregated with agg_func and assigned to it.
    'bagg' - all values in a sampling interval get aggregated with agg_func and the result gets assigned to the last
            grid point.
    'fagg' - all values in a sampling interval get aggregated with agg_func and the result gets assigned to the next
            grid point.


    Note, that. if possible, functions passed to agg_func will get projected internally onto pandas.resample methods,
    wich results in some reasonable performance boost - however, for this to work, you should pass functions that have
    the __name__ attribute initialised and the according methods name assigned to it.
    Furthermore, you shouldnt pass numpys nan-functions
    (nansum, nanmean,...) because those for example, have __name__ == 'nansum' and they will thus not trigger
    resample.func(), but the slower resample.apply(nanfunc). Also, internally, no nans get passed to the functions
    anyway, so that there is no point in passing the nan functions.

    Parameters
    ---------
    freq : Offset String
        The frequency of the grid you want to resample your data to.

    agg_func : function
        The function you want to use for aggregation.

    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceeding, succeeding or
        "surrounding" interval). See description above for more details.

    max_invalid_total_d : integer, default np.inf
        Maximum number of invalid (nan) datapoints, allowed per resampling interval. If max_invalid_total_d is
        exceeded, the interval gets resampled to nan. By default (np.inf), there is no bound to the number of nan values
        in an interval and only intervals containing ONLY nan values or those, containing no values at all,
        get projected onto nan

    max_invalid_consec_d : integer, default np.inf
        Maximum number of consecutive invalid (nan) data points, allowed per resampling interval.
        If max_invalid_consec_d is exceeded, the interval gets resampled to nan. By default (np.inf),
        there is no bound to the number of consecutive nan values in an interval and only intervals
        containing ONLY nan values, or those containing no values at all, get projected onto nan.

    max_invalid_total_f : integer, default np.inf
        Same as "max_invalid_total_d", only applying for the flags. The flag regarded as "invalid" value,
        is the one passed to empty_intervals_flag (default=flagger.BAD).
        Also this is the flag assigned to invalid/empty intervals.

    max_invalid_total_f : integer, default np.inf
        Same as "max_invalid_total_f", only applying onto flgas. The flag regarded as "invalid" value, is the one passed
        to empty_intervals_flag (default=flagger.BAD). Also this is the flag assigned to invalid/empty intervals.

    flag_agg_func : function, default: max
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).

    empty_intervals_flag : String, default None
        A Flag, that you want to assign to invalid intervals. Invalid are those intervals, that contain nan values only,
        or no values at all. Furthermore the empty_intervals_flag is the flag, serving as "invalid" identifyer when
        checking for "max_total_invalid_f" and "max_consec_invalid_f patterns". Default triggers flagger.BAD to be
        assigned.

    drop_flags : list or string, default None
        Flags that refer to values you want to drop before resampling - effectively excluding values that are flagged
        with a flag in drop_flags from the resampling process - this means that they also will not be counted in the
        the max_consec/max_total evaluation. Drop_flags = None results in NO flags being dropped initially.
    """


    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)
    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD

    drop_mask = dropper(field, drop_flags, flagger, [])
    datcol.drop(datcol[drop_mask].index, inplace=True)
    flagscol.drop(flagscol[drop_mask].index, inplace=True)

    datcol = aggregate2Freq(datcol, method, freq, agg_func, fill_value=np.nan,
                            max_invalid_total=max_invalid_total_d, max_invalid_consec=max_invalid_consec_d)
    flagscol = aggregate2Freq(flagscol, method, freq, flag_agg_func, fill_value=empty_intervals_flag,
                      max_invalid_total=max_invalid_total_f, max_invalid_consec=max_invalid_consec_f)

    # data/flags reshaping:
    data[field] = datcol
    reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, **kwargs)
    flagger = flagger.slice(drop=field).merge(reshaped_flagger)
    return data, flagger


@register
def proc_shift(data, field, flagger, freq, method, drop_flags=None, empty_intervals_flag=None, **kwargs):
    """
    Function to shift data points to regular (equidistant) timestamps.
    Values get shifted according to the keyword passed to 'method'.

    Note: all data nans get excluded defaultly from shifting. If drop_flags is None - all BAD flagged values get
    excluded as well.

    'nshift' -  every grid point gets assigned the nearest value in its range ( range = +/-(freq/2) )
    'bshift' -  every grid point gets assigned its first succeeding value - if there is one available in the
            succeeding sampling interval. (equals resampling wih "first")
    'fshift'  -  every grid point gets assigned its ultimately preceeding value - if there is one available in
            the preceeding sampling interval. (equals resampling with "last")


    Parameters
    ---------
    freq : Offset String
        The frequency of the grid you want to shift your data to.

    method: {'fagg', 'bagg', 'nagg'}, default 'nshift'
        Specifies if datapoints get propagated forwards, backwards or to the nearest grid timestamp. See function
        description for more details.

    empty_intervals_flag : String, default None
        A Flag, that you want to assign to grid points, where no values are avaible to be shifted to.
        Default triggers flagger.BAD to be assigned.

    drop_flags : list or string, default None
        Flags that refer to values you want to drop before shifting - effectively, excluding values that are flagged
        with a flag in drop_flags from the shifting process. Default - Drop_flags = None  - results in flagger.BAD
        values being dropped initially.

    """
    if data[field].empty:
        return data, flagger

    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)

    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD

    drop_mask = dropper(field, drop_flags, flagger, flagger.BAD)
    drop_mask |= datcol.isna()
    datcol[drop_mask] = np.nan
    datcol.dropna(inplace=True)
    flagscol.drop(drop_mask[drop_mask].index, inplace=True)

    datcol = shift2Freq(datcol, method, freq, fill_value=np.nan)
    flagscol = shift2Freq(flagscol, method, freq, fill_value=empty_intervals_flag)
    data[field] = datcol
    reshaped_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, **kwargs)
    flagger = flagger.slice(drop=field).merge(reshaped_flagger)
    return data, flagger


@register
def proc_transform(data, field, flagger, func, **kwargs):
    """
    Function to transform data columns with a transformation that maps series onto series of the same length.

    Note, that flags get preserved.

    Parameters
    ---------
    func : function
        Function to transform data[field] with.

    """
    data = data.copy()
    # NOTE: avoiding pd.Series.transform() in the line below, because transform does process columns element wise
    # (so interpolati   ons wouldn't work)
    new_col = pd.Series(func(data[field]), index=data[field].index)
    data[field] = new_col
    return data, flagger


@register
def proc_projectFlags(data, field, flagger, method, source, freq=None, drop_flags=None, **kwargs):

    """
    The Function projects flags of "source" onto flags of "field". Wherever the "field" flags are "better" then the
    source flags projected on them, they get overridden with this associated source flag value.
    Which "field"-flags are to be projected on which source flags, is controlled by the "method" and "freq"
    parameters.

    method: (field_flag in associated with "field", source_flags associated with "source")

    'inverse_nagg' - all field_flags within the range +/- freq/2 of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    'inverse_bagg' - all field_flags succeeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    'inverse_fagg' - all field_flags preceeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)

    'inverse_interpolation' - all field_flags within the range +/- freq of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)

    'inverse_nshift' - That field_flag within the range +/- freq/2, that is nearest to a source_flag, gets the source
        flags value. (if source_flag > field_flag)
    'inverse_bshift' - That field_flag succeeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    'inverse_nshift' - That field_flag preceeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)

    'match' - any field_flag with a timestamp matching a source_flags timestamp gets this source_flags value
    (if source_flag > field_flag)

    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you shoud pass the same drop flags keyword.

    Parameters
    ---------

    method: {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift'}
        The method used for projection of source flags onto field flags. See description above for more details.

    source: String
        The source source of flags projection.

    freq: Offset, default None
        The freq determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of source is used.

    drop_flags: list or String
        Flags referring to values that are to drop before flags projection. Relevant only when projecting wiht an
        inverted shift method. Defaultly flagger.BAD is listed.

    """
    flagscol = flagger.getFlags(source)
    if flagscol.empty:
        return data, flagger
    target_datcol = data[field]
    target_flagscol = flagger.getFlags(field)

    if (freq is None) and (method != 'match'):
        freq = pd.Timedelta(flagscol.index.freq)
        if freq is pd.NaT:
            raise ValueError(
                "Nor is {} a frequency regular timeseries, neither was a frequency passed to parameter 'freq'. "
                "Dont know what to do.".format(source)
            )
    if method[-13:] == 'interpolation':
        backprojected = flagscol.reindex(target_flagscol.index, method='bfill',
                                    tolerance=freq)
        fwrdprojected = flagscol.reindex(target_flagscol.index, method='ffill',
                                    tolerance=freq)
        b_replacement_mask = backprojected > target_flagscol
        target_flagscol.loc[b_replacement_mask] = backprojected.loc[b_replacement_mask]
        f_replacement_mask = (fwrdprojected > target_flagscol) & (fwrdprojected > backprojected)
        target_flagscol.loc[f_replacement_mask] = backprojected.loc[f_replacement_mask]

    if (method[-3:] == "agg") or (method == 'match'):
        # Aggregation - Inversion
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)

        flagscol = flagscol.reindex(target_flagscol.index, method=projection_method,
                                              tolerance=tolerance)
        replacement_mask = flagscol > target_flagscol
        target_flagscol.loc[replacement_mask] = flagscol.loc[replacement_mask]

    if method[-5:] == "shift":
        # NOTE: although inverting a simple shift seems to be a less complex operation, it has quite some
        # code assigned to it and appears to be more verbose than inverting aggregation -
        # that owes itself to the problem of BAD/invalid values blocking a proper
        # shift inversion and having to be outsorted before shift inversion and re-inserted afterwards.
        #
        # starting with the dropping and its memorization:

        drop_mask = dropper(field, drop_flags, flagger, flagger.BAD)
        drop_mask |= target_datcol.isna()
        target_flagscol_drops = target_flagscol[drop_mask]
        target_flagscol.drop(drop_mask[drop_mask].index, inplace=True)

        # shift inversion
        projection_method = METHOD2ARGS[method][0]
        tolerance = METHOD2ARGS[method][1](freq)
        flags_merged = pd.merge_asof(
            flagscol,
            pd.Series(target_flagscol.index.values,
                         index=target_flagscol.index,
                         name="pre_index"),
            left_index=True,
            right_index=True,
            tolerance=tolerance,
            direction=projection_method,
        )
        flags_merged.dropna(subset=["pre_index"], inplace=True)
        flags_merged = flags_merged.set_index(["pre_index"]).squeeze()

        # write flags to target
        replacement_mask = flags_merged > target_flagscol.loc[flags_merged.index]
        target_flagscol.loc[replacement_mask[replacement_mask].index] = flags_merged.loc[replacement_mask]

        # reinsert drops
        target_flagscol = target_flagscol.reindex(target_flagscol.index.join(target_flagscol_drops.index, how='outer'))
        target_flagscol.loc[target_flagscol_drops.index] = target_flagscol_drops.values

    flagger = flagger.setFlags(field=field, flag=target_flagscol, **kwargs)
    return data, flagger


@register
def proc_fork(data, field, flagger, suffix=ORIGINAL_SUFFIX, **kwargs):
    """
    The function generates a copy of the data "field" and inserts it under the name field + suffix into the existing
    data.

    Parameters
    ---------

    suffix: String
        Sub string to append to the forked data variables name.

    """
    fork_field = str(field) + suffix
    fork_dios = dios.DictOfSeries({fork_field: data[field]})
    fork_flagger = flagger.slice(drop=data.columns.drop(field)).rename(field, fork_field)
    data = mergeDios(data, fork_dios)
    flagger = flagger.merge(fork_flagger)
    return data, flagger


@register
def proc_drop(data, field, flagger, **kwargs):
    """
    The function drops field from the data dios and the flagger.
    """

    data = data[data.columns.drop(field)]
    flagger = flagger.slice(drop=field)
    return data, flagger


@register
def proc_rename(data, field, flagger, new_name, **kwargs):
    """
    The function renames field to new name (in both, the flagger and the data).
    """

    data.columns = mutateIndex(data.columns, field, new_name)
    new_flagger = flagger.rename(field, new_name)
    return data, new_flagger


def _drift_fit(x, shift_target, cal_mean):
    x_index = (x.index - x.index[0])
    x_data = x_index.total_seconds().values
    x_data = x_data / x_data[-1]
    y_data = x.values
    origin_mean = np.mean(y_data[:cal_mean])
    target_mean = np.mean(y_data[-cal_mean:])
    def modelWrapper(x, c, a=origin_mean, target_mean=target_mean):
        # final fitted curves val = target mean
        b = (target_mean - a) / (np.exp(c) - 1)
        return expModelFunc(x, a, b, c)
    dataFitFunc = functools.partial(modelWrapper, a=origin_mean, target_mean=target_mean)

    try:
        fitParas,_ = curve_fit(dataFitFunc, x_data, y_data, bounds=([0], [np.inf]))
        dataFit = dataFitFunc(x_data, fitParas[0])
        b_val = (shift_target - origin_mean) / (np.exp(fitParas[0]) - 1)
        dataShiftFunc = functools.partial(expModelFunc, a=origin_mean, b=b_val, c=fitParas[0])
        dataShift = dataShiftFunc(x_data)
    except RuntimeError:
        dataFit = np.array([0]*len(x_data))
        dataShift = np.array([0]*len(x_data))

    return dataFit, dataShift


@register
def proc_seefoExpDriftCorrecture(data, field, flagger, maint_data_field, cal_mean=5, flag_maint_period=False, **kwargs):
    """
    The function fits an exponential model to chunks of data[field].
    It is assumed, that between maintenance events, there is a drift effect shifting the meassurements in a way, that
    can be described by the model M:

    M(t, a, b, c) = a + b(exp(c*t))

    Where as the values y_0 and y_1, describing the mean value directly after the last maintenance event (y_0) and
    directly before the next maintenance event (y_1), impose the following additional conditions on the drift model:.

    M(0, a, b, c) = y0
    M(1, a, b, c) = y1

    Solving the equation, one obtains the one-parameter Model:

    M_drift(t, c) = y0 + [(y1 - y0)/(exp(c) - )] * (exp(c*t) - 1)

    For every datachunk in between maintenance events.

    After having found the optimal parameter c*, the correction is performed by bending the fitted curve M_drift(t, c*),
    in a way that it matches y2 at t=1 (,with y2 being the mean value observed directly after the end of the next
    maintenance event).
    This bended curve is given by:

    M_shift(t, c*) = M(t, y0, [(y1 - y0)/(exp(c*) - )], c*)

    And the new values at t are computed via:

    new_vals(t) = old_vals(t) + M_shift(t) - M_drift(t)

    Parameters
    ----------
    maint_data : pandas.Series
        The timeseries holding the maintenance information. The series' timestamp itself represent the beginning of a
        maintenance event, wheras the values represent the endings of the maintenance intervals
    cal_mean : Integer, default 5
        The number of values the mean is computed over, for obtaining the value level directly after and
        directly before maintenance event. This values are needed for shift calibration. (see above description)
    flag_maint_period : bool, default False
        Wheather or not to flag BAD the values directly obtained while maintenance.

    """

    # 1: extract fit intervals:
    if data[maint_data_field].empty:
        return data, flagger
    data = data.copy()
    to_correct = data[field]
    drift_frame = pd.DataFrame({'drift_group': np.nan, to_correct.name: to_correct.values}, index=to_correct.index)
    maint_data = data[maint_data_field]
    # group the drift frame
    for k in range(0, maint_data.shape[0]-1):
        # assign group numbers for the timespans in between one maintenance ending and the beginning of the next
        # maintenance time itself remains np.nan assigned
        drift_frame.loc[maint_data.values[k]:pd.Timestamp(maint_data.index[k+1]), 'drift_group'] = k
    drift_grouper = drift_frame.groupby('drift_group')
    # define target values for correction
    shift_targets = drift_grouper.aggregate(lambda x: x[:cal_mean].mean()).shift(-1)

    for k, group in drift_grouper:
        dataSeries = group[to_correct.name]
        dataFit, dataShiftTarget = _drift_fit(dataSeries, shift_targets.loc[k, :][0], cal_mean)
        dataFit = pd.Series(dataFit, index=group.index)
        dataShiftTarget = pd.Series(dataShiftTarget, index=group.index)
        dataShiftVektor = dataShiftTarget - dataFit
        shiftedData = dataSeries + dataShiftVektor
        to_correct[shiftedData.index] = shiftedData

    if flag_maint_period:
        to_flag = drift_frame['drift_group']
        to_flag = to_flag.drop(to_flag[:maint_data.index[0]].index)
        to_flag = to_flag[to_flag.isna()]
        flagger = flagger.setFlags(field, loc=to_flag, **kwargs)
    return data, flagger