#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.core.register import register
from saqc.lib.ts_operators import interpolateNANs, aggregate2Freq, shift2Freq
from saqc.lib.tools import toSequence


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
                            downgrade_interpolation=False, **kwargs):

    datcol = data[field].copy()
    flagscol = flagger.getFlags(field)
    drop_flags = toSequence(drop_flags)
    drop_mask = pd.Series(False, index=datcol.index)
    for f in drop_flags:
        drop_mask |= flagger.isFlagged(field, flag=f)
    datcol[drop_mask] = np.nan
    datcol.dropna()
    # account for annoying case of subsequent frequency aligned values, differing exactly by the margin
    # 2*freq:
    spec_case_mask = datcol.index.to_series()
    spec_case_mask = spec_case_mask - spec_case_mask.shift(1)
    spec_case_mask = spec_case_mask == 2 * pd.Timedelta(freq)
    spec_case_mask = spec_case_mask[spec_case_mask]
    spec_case_mask = spec_case_mask.resample(freq).asfreq().dropna()

    if not spec_case_mask.empty:
        spec_case_mask = spec_case_mask.tshift(-1, freq)

    grid_index = pd.date_range(start=data.index[0].floor(freq), end=data.index[-1].ceil(freq), freq=freq,
                               name=data.index.name)

    data.reindex(
        data.index.join(grid_index, how="outer", )
    )

    inter_data, chunk_bounds = interpolateNANs(
        datcol, method, order=inter_order, inter_limit=2, downgrade_interpolation=downgrade_interpolation,
        return_chunk_bounds=True
    )

    # exclude falsely interpolated values:
    data[spec_case_mask.index] = np.nan
    data = data.asfreq(freq)
    data[field] = inter_data

    # reshape flagger (tiny hack to resample with overlapping intervals):
    flagscol.drop(flagscol[drop_mask].index, inplace=True)
    flagscol2 = flagscol.copy()
    flagscol2.index = flagscol.index.shift(freq=pd.Timedelta(freq))
    max_ser1 = flagscol.resample(2*pd.Timedelta(freq)).max()
    max_ser2 = flagscol2.resample(2*pd.Timedelta(freq)).max()
    max_ser1.index = max_ser1.index.shift(freq=pd.Timedelta(freq))
    flagscol = max_ser1.align(max_ser2)[0]
    flagscol[max_ser2.index] = max_ser2
    flagger_new = flagger.initFlags(inter_data).setFlags(field, flag=flagscol, force=True, **kwargs)

    # block chunk ends of interpolation
    flags_to_block = pd.Series(np.nan, index=chunk_bounds).astype(flagger_new.dtype)
    flagger_new = flagger_new.setFlags(field, loc=chunk_bounds, flag=flags_to_block, force=True)

    flagger_new = flagger.slice(drop=field).merge(flagger_new)
    return data, flagger_new


@register
def proc_resample(data, field, flagger, freq, func=np.mean, max_invalid_total_d=np.inf, max_invalid_consec_d=np.inf,
                  max_invalid_consec_f=np.inf, max_invalid_total_f=np.inf, flag_agg_func=max, method='bagg', **kwargs):

    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)


    if func == "shift":
        datcol = shift2Freq(datcol, method, freq, fill_value=np.nan)
        flagscol = shift2Freq(flagscol, method, freq, fill_value=flagger.BAD)

    else:
        datcol = aggregate2Freq(datcol, method, freq, func, fill_value=np.nan,
                          max_invalid_total=max_invalid_total_d, max_invalid_consec=max_invalid_consec_d)
        flagscol = aggregate2Freq(flagscol, method, freq, flag_agg_func, fill_value=flagger.BAD,
                          max_invalid_total=max_invalid_total_f, max_invalid_consec=max_invalid_consec_f)

    # data/flags reshaping:
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