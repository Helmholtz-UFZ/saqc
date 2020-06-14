#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from saqc.core.register import register
from saqc.lib.ts_operators import polyRoller, polyRollerNoMissing, polyRoller_numba, polyRollerNoMissing_numba, \
    validationAgg


@register
def modelling_polyFit(data, field, flagger, winsz, polydeg, numba='auto', eval_flags=True, min_periods=0, **kwargs):
    data = data.copy()
    to_fit = data[field]
    flags = flagger.getFlags(field)
    if numba == 'auto':
        if to_fit.shape[0] < 200000:
            numba = False
        else:
            numba = True

    val_range = np.arange(0, winsz)
    center_index = np.floor(winsz / 2)
    if min_periods < winsz:
        if min_periods > 0:
            max_nan_total = winsz - min_periods
            to_fit = to_fit.rolling(winsz, center=True).apply(validationAgg, raw=True, args=(max_nan_total))
        # we need a missing value marker that is not nan, because nan values dont get passed by pandas rolling method
        miss_marker = to_fit.min()
        miss_marker = np.floor(miss_marker - 1)
        na_mask = to_fit.isna()
        to_fit[na_mask] = miss_marker
        if numba:
            residues = to_fit.rolling(winsz, center=True).apply(polyRoller_numba, args=(miss_marker, val_range,
                                                                                    center_index, polydeg),
                                                    raw=True, engine='numba', engine_kwargs={'no_python': True})
        else:
            residues = to_fit.rolling(winsz, center=True).apply(polyRoller,
                                                            args=(miss_marker, val_range, center_index, polydeg), raw=True)
        residues = residues - to_fit
        residues[na_mask] = np.nan
    else:
        # we only fit fully populated intervals:
        if numba:
            residues = to_fit.rolling(winsz, center=True).apply(polyRollerNoMissing_numba, args=(val_range,
                                                                                        center_index, polydeg),
                                                                engine='numba', engine_kwargs={'no_python': True}, raw=True)
        else:
            residues = to_fit.rolling(winsz, center=True).apply(polyRollerNoMissing,
                                                                args=(val_range, center_index, polydeg), raw=True)

    data[field] = residues
    if eval_flags:
        num_cats, codes = flags.factorize()
        num_cats = pd.Series(num_cats, index=flags.index).rolling(winsz, center=True, min_periods=min_periods).max()
        nan_samples = num_cats[num_cats.isna()]
        num_cats.drop(nan_samples.index, inplace=True)
        to_flag = pd.Series(codes[num_cats.astype(int)], index=num_cats.index)
        to_flag = to_flag.align(nan_samples)[0]
        to_flag[nan_samples.index] = flags[nan_samples.index]
        flagger = flagger.setFlags(field, flags=to_flag, **kwargs)

    return data, flagger
