#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.funcs.register import register
from saqc.lib.ts_operators import interpolateNANs, validationTrafo, aggregate2Freq
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
def proc_resample(data, field, flagger, freq, func="mean", max_invalid_total_d=np.inf, max_invalid_consec_d=np.inf,
                  max_invalid_consec_f=np.inf, max_invalid_total_f=np.inf, flag_agg_func='max', method='bagg', **kwargs):

    data = data.copy()
    datcol = data[field]
    flagscol = flagger.getFlags(field)

    func = composeFunction(func)
    flag_agg_func = composeFunction(flag_agg_func)

    if func == 'shift':
        datcol = shift2Freq(datcol, method, freq, fill_value=fill_value)

    # data resampling
    datcol = aggregate2Freq(datcol, method, agg_func=func, freq=freq, fill_value=np.nan,
                          max_invalid_total=max_invalid_total_d, max_invalid_consec=max_invalid_consec_d)

    # flags resampling:
    flagscol = aggregate2Freq(flagscol, method, agg_func=flag_agg_func, freq=freq, fill_value=flagger.BAD,
                          max_invalid_total=max_invalid_total_f, max_invalid_consec=max_invalid_consec_f)

    # data/flags reshaping:
    data[field] = datcol
    reshape_flagger = flagger.initFlags(datcol).setFlags(field, flag=flagscol, force=True, **kwargs)
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