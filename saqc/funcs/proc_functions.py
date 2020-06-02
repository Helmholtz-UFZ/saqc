#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.core.register import register
from saqc.lib.ts_operators import interpolateNANs, aggregate2Freq, shift2Freq
from saqc.lib.tools import toSequence

METHOD2ARGS = {'inverse_fshift': ('backward', pd.Timedelta),
               'inverse_bshift': ('forward', pd.Timedelta),
               'inverse_nshift': ('nearest', lambda x: pd.Timedelta(x)/2),
               'inverse_fagg': ('bfill', pd.Timedelta),
               'inverse_bagg': ('ffill', pd.Timedelta),
               'inverse_nagg': ('nearest', lambda x: pd.Timedelta(x)/2)}


@register
def proc_interpolateMissing(data, field, flagger, method, inter_order=2, inter_limit=2, interpol_flag='UNFLAGGED',
                            downgrade_interpolation=False, return_chunk_bounds=False, not_interpol_flags=None, **kwargs):


    data = data.copy()
    inter_data = interpolateNANs(data[field], method, order=inter_order, inter_limit=inter_limit,
                           downgrade_interpolation=downgrade_interpolation, return_chunk_bounds=return_chunk_bounds)
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

    datcol = data[field].copy()
    flagscol = flagger.getFlags(field)
    if drop_flags is None:
        drop_flags = flagger.BAD
    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD
    drop_flags = toSequence(drop_flags)
    drop_mask = flagscol.isna()
    for f in drop_flags:
        drop_mask |= flagger.isFlagged(field, flag=f)
    drop_mask |= datcol.isna()
    datcol[drop_mask] = np.nan
    datcol.dropna(inplace=True)

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
def proc_resample(data, field, flagger, freq, func=np.mean, max_invalid_total_d=np.inf, max_invalid_consec_d=np.inf,
                  max_invalid_consec_f=np.inf, max_invalid_total_f=np.inf, flag_agg_func=max, method='bagg',
                  empty_intervals_flag=None, **kwargs):

    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)
    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD

    datcol = aggregate2Freq(datcol, method, freq, func, fill_value=np.nan,
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
    # Note: all data nans get excluded defaultly from shifting. If drop_flags is None - all BAD flagged values get
    # excluded as well.
    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)

    if empty_intervals_flag is None:
        empty_intervals_flag = flagger.BAD
    if drop_flags is None:
        drop_flags = flagger.BAD

    drop_flags = toSequence(drop_flags)
    drop_mask = pd.Series(False, index=datcol.index)
    for f in drop_flags:
        drop_mask |= flagger.isFlagged(field, flag=f)
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
    data = data.copy()
    # NOTE: avoiding pd.Series.transform() in the line below, because transform does process columns element wise
    # (so interpolations wouldn't work)
    new_col = pd.Series(func(data[field]), index=data[field].index)
    data[field] = new_col
    return data, flagger

@register
def proc_projectFlags(data, field, flagger, target, method, freq=None, drop_flags=None, **kwargs):
    datcol = data[field].copy()
    target_datcol = data[target].copy()
    flagscol = flagger.getFlags(field)
    target_flagscol = flagger.getFlags(target)

    if freq is None:
        freq = pd.Timedelta(datcol.index.freq)
        if freq is pd.NaT:
            raise ValueError(
                "Nor is {} a frequency regular timeseries, neither was a frequency passed to parameter 'freq'. "
                "Dont know what to do.".format(field)
            )
    if method[-3:] == "agg":
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
        # that ownes itself to the problem of BAD/invalid values blocking a proper
        # shift inversion and having to be outsorted before shift inversion and re-inserted afterwards.
        #
        # starting with the dropping and its memorization:
        if drop_flags is None:
            drop_flags = flagger.BAD

        drop_flags = toSequence(drop_flags)
        drop_mask = pd.Series(False, index=target_datcol.index)
        for f in drop_flags:
            drop_mask |= flagger.isFlagged(field, flag=f)
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

    flagger = flagger.setFlags(field=target, flag=target_flagscol.values)
    return data, flagger