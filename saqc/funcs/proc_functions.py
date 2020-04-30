#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.funcs.register import register
from saqc.lib.ts_operators import interpolateNANs, validationTrafo
from saqc.lib.tools import composeFunction, toSequence


@register()
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


@register()
def proc_resample(data, field, flagger, freq, func="mean", max_invalid_total=None, max_invalid_consec=None,
                  flag_agg_func='max', **kwargs):
    data = data.copy()
    datcol = data[field]
    d_start = datcol.index[0].floor(freq)
    d_end = datcol.index[-1].ceil(freq)

    # filter data for invalid patterns
    if (max_invalid_total is not None) | (max_invalid_consec is not None):
        if not max_invalid_total:
            max_invalid_total = np.inf
        if not max_invalid_consec:
            max_invalid_consec = np.inf

        datcol = datcol.groupby(pd.Grouper(freq=freq)).transform(validationTrafo, max_nan_total=max_invalid_total,
                                                             max_nan_consec=max_invalid_consec)
    nanmask = np.isnan(datcol)
    datcol = datcol[~nanmask]
    datflags = flagger.getFlags()[field]
    datflags = datflags[~nanmask]
    datresampler = datcol.resample(freq)
    flagsresampler = datflags.resample(freq)

    # data resampling:
    try:
        datcol = getattr(datresampler, func)()
    except AttributeError:
        func = composeFunction(func)
        datcol = datresampler.apply(func)

    # flags resampling:
    try:
        datflags = getattr(flagsresampler, flag_agg_func)()
    except AttributeError:
        flag_agg_func = composeFunction(flag_agg_func)
        datflags = flagsresampler.apply(flag_agg_func)

    # insert freqgrid (for consistency reasons -> in above step, start and ending chunks can get lost due to invalid
    # intervals):
    grid = pd.date_range(d_start, d_end, freq=freq)
    datcol = datcol.reindex(grid)
    datflags = datflags.reindex(grid)
    # data/flags reshaping:
    data[field] = datcol
    reshape_flagger = flagger.initFlags(datcol).setFlags(field, flag=datflags, force=True, **kwargs)
    flagger = flagger.getFlagger(drop=field).setFlagger(reshape_flagger)
    return data, flagger


@register()
def proc_transform(data, field, flagger, func, **kwargs):
    data = data.copy()
    func = composeFunction(func)
    # NOTE: avoiding pd.Series.transform() in the line below, because transform does process columns element wise
    # (so interpolations wouldn't work)
    new_col = pd.Series(func(data[field]), index=data[field].index)
    data[field] = new_col
    return data, flagger