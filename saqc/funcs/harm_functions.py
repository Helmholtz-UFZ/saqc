#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import logging
from saqc.core.register import register
from saqc.funcs.proc_functions import proc_interpolateGrid, proc_shift, proc_fork, proc_resample, proc_projectFlags, \
    proc_drop, proc_rename, ORIGINAL_SUFFIX


logger = logging.getLogger("SaQC")

# some wrapper functions to mimicking classic harmonization look and feel


@register
def harm_shift2Grid(data, field, flagger, freq, method="nshift", drop_flags=None, empty_intervals_flag=None, **kwargs):

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_shift(data, field, flagger, freq, method, drop_flags=drop_flags,
                               empty_intervals_flag=empty_intervals_flag, **kwargs)
    return data, flagger


@register
def harm_aggregate2Grid(
    data, field, flagger, freq, value_func, flag_func=np.nanmax, method="nagg", drop_flags=None,
        empty_intervals_flag=None, **kwargs
):

    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_resample(data, field, flagger, freq, agg_func=value_func, flag_agg_func=flag_func,
                                  method=method, empty_intervals_flag=empty_intervals_flag, drop_flags=drop_flags,
                                  **kwargs)
    return data, flagger


@register
def harm_linear2Grid(data, field, flagger, freq, drop_flags=None, empty_intervals_flag=None, **kwargs):
    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_interpolateGrid(data, field, flagger, freq, 'time',
                                         drop_flags=drop_flags, empty_intervals_flag=empty_intervals_flag, **kwargs)
    return data, flagger


@register
def harm_interpolate2Grid(
    data, field, flagger, freq, method, order=1, drop_flags=None, empty_intervals_flag=None, **kwargs,
):
    data, flagger = proc_fork(data, field, flagger)
    data, flagger = proc_interpolateGrid(data, field, flagger, freq, method=method, inter_order=order,
                                         drop_flags=drop_flags, empty_intervals_flag=empty_intervals_flag,
                                         **kwargs)
    return data, flagger


@register
def harm_deharmonize(
    data, field, flagger, method, drop_flags=None, **kwargs
):

    data, flagger = proc_projectFlags(data, str(field) + ORIGINAL_SUFFIX, flagger, method, source=field,
                                      drop_flags=drop_flags,
                                      **kwargs)
    data, flagger = proc_drop(data, field, flagger)
    data, flagger = proc_rename(data, str(field) + ORIGINAL_SUFFIX, flagger, field)
    return data, flagger
