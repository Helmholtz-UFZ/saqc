#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.funcs.register import register
from saqc.lib.ts_operators import interpolateNANs, validationTrafo
from saqc.lib.tools import composeFunction

@register()
def proc_interpolateMissing(data, field, flagger, method, inter_order=2, inter_limit=2, interpol_flag='UNFLAGGED',
                            downgrade_interpolation=False, return_chunk_bounds=False, **kwargs):

    inter_data = interpolateNANs(data[field], method, order=inter_order, inter_limit=inter_limit,
                           downgrade_interpolation=downgrade_interpolation, return_chunk_bounds=return_chunk_bounds)
    interpolated = data[field].isna() & inter_data.notna()

    if interpol_flag:
        flagger = flagger.setFlags(field, loc=interpolated[interpolated].index, force=True,
                                   flag=getattr(flagger, interpol_flag), **kwargs)
    return inter_data, flagger

@register()
def proc_resample(data, field, flagger, freq, func="mean", max_invalid_total=None, max_invalid_consec=None,
                  flag_agg_func='max', **kwargs):
    datcol = data[field]

    # filter data for invalid patterns
    if (max_invalid_total is None) | (max_invalid_consec is None):
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

    # data/flags reshaping:
    data[field] = datcol
    all_flags = flagger.getFlags()
    all_flags[field] = datflags
    flagger = flagger.initFlags(flags=all_flags)

    return data, flagger