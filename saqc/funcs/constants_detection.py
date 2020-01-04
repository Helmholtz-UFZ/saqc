#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from saqc.funcs.register import register
from saqc.lib.statistic_functions import varQC
from saqc.lib.tools import (
    valueRange,
    slidingWindowIndices,
    retrieveTrustworthyOriginal,
    groupConsecutives
)


@register("constant")
def flagConstant(data, field, flagger, eps, window, **kwargs):

    window = pd.tseries.frequencies.to_offset(window)

    col = data[field]
    index = col.index
    flagger_mask = pd.Series(data=np.zeros_like(col), index=index, name=field, dtype=bool)

    for srs in groupConsecutives(col.diff().abs().le(eps)):
        if np.all(srs):
            start = index[index.get_loc(srs.index[0]) - 1]
            stop = srs.index[-1]
            if stop - start >= window:
                flagger_mask[start:stop] = True

    flagger = flagger.setFlags(field, flagger_mask, **kwargs)
    return data, flagger


@register("constant_varianceBased")
def flagConstant_varianceBased(
    data,
    field,
    flagger,
    plateau_window_min="12h",
    plateau_var_limit=0.0005,
    var_total_nans=np.inf,
    var_consec_nans=np.inf,
    **kwargs
):

    """Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:

    (1) n > "plateau_interval_min"
    (2) variance(y(t),...,y(t+n) < plateau_var_limit

    :param data:                        The pandas dataframe holding the data-to-be flagged.
                                        Data must be indexed by a datetime series and be harmonized onto a
                                        time raster with seconds precision (skips allowed).
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

    dataseries, data_rate = retrieveTrustworthyOriginal(data, field, flagger)

    min_periods = int(np.ceil(pd.Timedelta(plateau_window_min) / pd.Timedelta(data_rate)))

    plateaus = dataseries.rolling(
        window=plateau_window_min, min_periods=min_periods
    ).apply(
        lambda x: True
        if varQC(x, var_total_nans, var_consec_nans) < plateau_var_limit
        else np.nan,
        raw=False,
    )

    # are there any candidates for beeing flagged plateau-ish
    if plateaus.sum() == 0:
        return data, flagger

    plateaus.fillna(method="bfill", limit=min_periods-1, inplace=True)

    # result:
    plateaus = (plateaus[plateaus == 1.0]).index

    flagger = flagger.setFlags(field, plateaus, **kwargs)
    return data, flagger
