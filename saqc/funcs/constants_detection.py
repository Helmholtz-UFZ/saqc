#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging


import numpy as np
import pandas as pd
from .register import register


from .statistic_functions import (
    var_qc,
    mean_qc)

from ..lib.tools import (
    valueRange,
    slidingWindowIndices,
    retrieveTrustworthyOriginal,
    getPandasVarNames,
    getPandasData,
    offset2periods,
    checkQCParameters)



@register("constant")
def flagConstant(data, flags, field, flagger, eps, length, thmin=None, **kwargs):
    datacol = data[field]
    flagcol = flags[field]

    length = ((pd.to_timedelta(length) - data.index.freq)
              .to_timedelta64()
              .astype(np.int64))

    values = (datacol
              .mask((datacol < thmin) | datacol.isnull())
              .values
              .astype(np.int64))

    dates = datacol.index.values.astype(np.int64)

    mask = np.isfinite(values)

    for start_idx, end_idx in slidingWindowIndices(datacol.index, length):
        mask_chunk = mask[start_idx:end_idx]
        values_chunk = values[start_idx:end_idx][mask_chunk]
        dates_chunk = dates[start_idx:end_idx][mask_chunk]

        # we might have removed dates from the start/end of the
        # chunk resulting in a period shorter than 'length'
        # print (start_idx, end_idx)
        if valueRange(dates_chunk) < length:
            continue
        if valueRange(values_chunk) < eps:
            flagcol[start_idx:end_idx] = flagger.setFlags(flagcol[start_idx:end_idx], **kwargs)

    data[field] = datacol
    flags[field] = flagcol
    return data, flags


@register("constants_varianceBased")
def flagConstants_VarianceBased(data, flags, field, flagger, plateau_window_min='12h', plateau_var_limit=0.0005,
                                var_total_nans=np.inf, var_consec_nans=np.inf, **kwargs):

    """Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:

    (1) n > "plateau_interval_min"
    (2) variance(y(t),...,y(t+n) < plateau_var_limit

    :param data:                        The pandas dataframe holding the data-to-be flagged.
                                        Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision (skips allowed).
    :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
    :param field:                       Fieldname of the Soil moisture measurements field in data.
    :param flagger:                     A flagger - object. (saqc.flagger.X)
    :param plateau_window_min:          Offset String. Only intervals of minimum size "plateau_window_min" have the
                                        chance to get flagged as constant intervals
    :param plateau_var_limit:           Float. The upper barrier, the variance of an interval mus not exceed, if the
                                        interval wants to be flagged a plateau.
    :param var_total_nans:              maximum number of nan values tolerated in an interval, for retrieving a valid
                                        variance from it. (Intervals with a number of nans exceeding "var_total_nans"
                                        have no chance to get flagged a plateau!)
    :param var_consec_nans:            Maximum number of consecutive nan values allowed in an interval to retrieve a
                                        valid  variance from it. (Intervals with a number of nans exceeding
                                        "var_total_nans" have no chance to get flagged a plateau!)
    """

    para_check_1 = checkQCParameters({'data': {'value': data,
                                               'type': [pd.Series, pd.DataFrame],
                                               'tests': {'harmonized': lambda x: pd.infer_freq(x.index) is not None}},
                                      'flags': {'value': flags,
                                                'type': [pd.Series, pd.DataFrame]},
                                      'field': {'value': field,
                                                'type': [str],
                                                'tests': {'scheduled in data':
                                                          lambda x: x in getPandasVarNames(data)}}},
                                     kwargs['func_name'])

    dataseries, data_rate = retrieveTrustworthyOriginal(data, flags, field, flagger)

    para_check_2 = checkQCParameters({'plateau_window_min': {'value': plateau_window_min,
                                                             'type': [str],
                                                             'tests': {'Valid Offset String':
                                                                       lambda x: pd.Timedelta(x).total_seconds() % 1
                                                                                 == 0}},
                                      'plateau_var_limit': {'value': plateau_var_limit,
                                                            'type': [int, float],
                                                            'range': [0, np.inf]},
                                      'data_rate':          {'value': data_rate,
                                                             'tests': {'not nan': lambda x: x is not np.nan}},
                                      'var_total_nans': {'value': var_total_nans,
                                                         'type': [int, float],
                                                         'range': [0, np.inf]},
                                      'var_consec_nans': {'value': var_consec_nans,
                                                          'type': [int, float],
                                                          'range': [0, np.inf]}
                                      },
                                     kwargs['func_name'])

    if (para_check_1 < 0) | (para_check_2 < 0):
        logging.warning('test {} will be skipped because not all input parameters satisfied '
                        'the requirements'.format(kwargs['func_name']))
        return data, flags

    min_periods = int(offset2periods(plateau_window_min, data_rate))

    plateaus = dataseries.rolling(window=plateau_window_min, min_periods=min_periods).apply(
        lambda x: True if var_qc(x, var_total_nans, var_consec_nans) < plateau_var_limit else np.nan, raw=False)

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flags

    plateaus.fillna(method='bfill', limit=min_periods, inplace=True)

    # result:
    plateaus = (plateaus[plateaus == 1.0]).index

    flags = flagger.setFlags(flags, field, plateaus, **kwargs)
    return data, flags
