#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import logging
from saqc.core.register import register
from saqc.funcs.proc_functions import (
    proc_interpolateGrid,
    proc_shift,
    proc_fork,
    proc_resample,
    proc_projectFlags,
    proc_drop,
    proc_rename,
    ORIGINAL_SUFFIX,
)


logger = logging.getLogger("SaQC")

@register(all_data=False)
def harm_shift2Grid(data, field, flagger, freq, method="nshift", to_drop=None, **kwargs):

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_shift(
        data, field, flagger, freq, method, to_drop=to_drop, empty_intervals_flag=flagger.UNFLAGGED, **kwargs
    )
    return data, flagger


@register(all_data=False)
def harm_aggregate2Grid(
    data, field, flagger, freq, value_func, flag_func=np.nanmax, method="nagg", to_drop=None, **kwargs
):

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_resample(
        data,
        field,
        flagger,
        freq,
        agg_func=value_func,
        flag_agg_func=flag_func,
        method=method,
        empty_intervals_flag=flagger.UNFLAGGED,
        to_drop=to_drop,
        all_na_2_empty=True,
        **kwargs,
    )
    return data, flagger


@register(all_data=False)
def harm_linear2Grid(data, field, flagger, freq, to_drop=None, **kwargs):
    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_interpolateGrid(
        data, field, flagger, freq, "time", to_drop=to_drop, empty_intervals_flag=flagger.UNFLAGGED, **kwargs
    )
    return data, flagger


@register(all_data=False)
def harm_interpolate2Grid(
    data, field, flagger, freq, method, order=1, to_drop=None, **kwargs,
):
    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_interpolateGrid(
        data,
        field,
        flagger,
        freq,
        method=method,
        inter_order=order,
        to_drop=to_drop,
        empty_intervals_flag=flagger.UNFLAGGED,
        **kwargs,
    )
    return data, flagger


@register(all_data=False)
def harm_deharmonize(data, field, flagger, method, to_drop=None, **kwargs):

    data, flagger = proc_projectFlags(
        data, str(field) + ORIGINAL_SUFFIX, flagger, method, source=field, to_drop=to_drop, **kwargs
    )
    data, flagger = proc_drop(data, field, flagger)
    data, flagger = proc_rename(data, str(field) + ORIGINAL_SUFFIX, flagger, field)
    return data, flagger
