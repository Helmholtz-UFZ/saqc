#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dios

from scipy.signal import savgol_filter

from saqc.core.register import register
from saqc.lib.tools import retrieveTrustworthyOriginal, detectDeviants


@register(masking='all')
def breaks_flagRegimeAnomaly(data, field, flagger, cluster_field, norm_spread, linkage_method='single',
                             metric=lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y)),
                             norm_frac=0.5, set_cluster=True, set_flags=True, **kwargs):
    """
    A function to flag values belonging to an anomalous regime regarding modelling regimes of field.

    "Normality" is determined in terms of a maximum spreading distance, regimes must not exceed in respect
    to a certain metric and linkage method.

    In addition, only a range of regimes is considered "normal", if it models more then `norm_frac` percentage of
    the valid samples in "field".

    Note, that you must detect the regime changepoints prior to calling this function.

    Note, that it is possible to perform hypothesis tests for regime equality by passing the metric
    a function for p-value calculation and selecting linkage method "complete".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger
        A flagger object, holding flags and additional Informations related to `data`.
    cluster_field : str
        The name of the column in data, holding the cluster labels for the samples in field. (has to be indexed
        equal to field)
    norm_spread : float
        A threshold denoting the valuelevel, up to wich clusters a agglomerated.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the variables.
    metric : Callable[[numpy.array, numpy.array], float], default lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes. Defaults to just the difference in mean.
    norm_frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.
    set_cluster : bool, default True
        If True, all data, considered "anormal", gets assigned a negative clusterlabel.
    set_flags : bool, default True
        Wheather or not to flag abnormal values (do not flag them, if you want to correct them
        afterwards, becasue flagged values usually are not visible in further tests.).

    kwargs

    Returns
    -------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    """

    clusterser = data[cluster_field]
    cluster = np.unique(clusterser)
    cluster_dios = dios.DictOfSeries({i: data[field][clusterser == i] for i in cluster})
    plateaus = detectDeviants(cluster_dios, metric, norm_spread, norm_frac, linkage_method, 'samples')

    if set_flags:
        for p in plateaus:
            flagger = flagger.setFlags(field, loc=cluster_dios.iloc[:, p].index, **kwargs)

    if set_cluster:
        for p in plateaus:
            if cluster[p] > 0:
                clusterser[clusterser == cluster[p]] = -cluster[p]

    data[cluster_field] = clusterser
    return data, flagger


@register(masking='field')
def breaks_flagSpektrumBased(
    data,
    field,
    flagger,
    thresh_rel=0.1,
    thresh_abs=0.01,
    first_der_factor=10,
    first_der_window="12h",
    scnd_der_ratio_range=0.05,
    scnd_der_ratio_thresh=10,
    smooth=True,
    smooth_window=None,
    smooth_poly_deg=2,
    **kwargs
):

    """
    The Function is a generalization of the Spectrum based break flagging mechanism as presented in:

    The function flags breaks (jumps/drops) in input measurement series by evaluating its derivatives.
    A measurement y_t is flagged a, break, if:

    (1) y_t is changing relatively to its preceeding value by at least (100*`rel_change_rate_min`) percent
    (2) y_(t-1) is difffering from its preceeding value, by a margin of at least `thresh_abs`
    (3) Absolute first derivative |(y_t)'| has to be at least `first_der_factor` times as big as the arithmetic middle
        over all the first derivative values within a 2 times `first_der_window_size` hours window, centered at t.
    (4) The ratio of the second derivatives at t and t+1 has to be "aproximately" 1.
        ([1-`scnd_der_ration_margin_1`, 1+`scnd_ratio_margin_1`])
    (5) The ratio of the second derivatives at t+1 and t+2 has to be larger than `scnd_der_ratio_margin_2`

    NOTE 1: As no reliable statement about the plausibility of the meassurements before and after the jump is possible,
    only the jump itself is flagged. For flagging constant values following upon a jump, use a flagConstants test.

    NOTE 2: All derivatives in the reference publication are obtained by applying a Savitzky-Golay filter to the data
    before differentiating.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    thresh_rel : float, default 0.1
        Float in [0,1]. See (1) of function description above to learn more
    thresh_abs : float, default 0.01
        Float > 0. See (2) of function descritpion above to learn more.
    first_der_factor : float, default 10
        Float > 0. See (3) of function descritpion above to learn more.
    first_der_window_range : str, default '12h'
        Offset string. See (3) of function description to learn more.
    scnd_der_ratio_margin_1 : float, default 0.05
        Float in [0,1]. See (4) of function descritpion above to learn more.
    scnd_der_ratio_margin_2 : float, default 10
        Float in [0,1]. See (5) of function descritpion above to learn more.
    smooth : bool, default True
        Method for obtaining dataseries' derivatives.
        * False: Just take series step differences (default)
        * True: Smooth data with a Savitzky Golay Filter before differentiating.
    smooth_window : {None, str}, default 2
        Effective only if `smooth` = True
        Offset string. Size of the filter window, used to calculate the derivatives.
    smooth_poly_deg : int, default 2
        Effective only, if `smooth` = True
        Polynomial order, used for smoothing with savitzk golay filter.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    The Function is a generalization of the Spectrum based break flagging mechanism as presented in:

    [1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.

    Find a brief mathematical description of the function here:

    [2] https://git.ufz.de/rdm-software/saqc/-/blob/testfuncDocs/docs/funcs
        /FormalDescriptions.md#breaks_flagspektrumbased
    """

    # retrieve data series input at its original sampling rate
    dataseries, data_rate = retrieveTrustworthyOriginal(data, field, flagger)

    if smooth_window is None:
        smooth_window = 3 * pd.Timedelta(data_rate)
    else:
        smooth_window = pd.Timedelta(smooth_window)

    # relative - change - break criteria testing:
    abs_change = np.abs(dataseries.shift(+1) - dataseries)
    breaks = (abs_change > thresh_abs) & (abs_change / dataseries > thresh_rel)
    breaks = breaks[breaks]

    # First derivative criterion
    smoothing_periods = int(np.ceil((smooth_window.seconds / data_rate.n)))
    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for brake in breaks.index:
        # slice out slice-to-be-filtered (with some safety extension of 12 times the data rate)
        slice_start = brake - pd.Timedelta(first_der_window) - smoothing_periods * pd.Timedelta(data_rate)
        slice_end = brake + pd.Timedelta(first_der_window) + smoothing_periods * pd.Timedelta(data_rate)
        data_slice = dataseries[slice_start:slice_end]

        # obtain first derivative:
        if smooth is True:
            first_deri_series = pd.Series(
                data=savgol_filter(data_slice, window_length=smoothing_periods, polyorder=smooth_poly_deg, deriv=1,),
                index=data_slice.index,
            )
        else:
            first_deri_series = data_slice.diff()

        # condition constructing and testing:
        test_slice = first_deri_series[brake - pd.Timedelta(first_der_window) : brake + pd.Timedelta(first_der_window)]

        test_sum = abs((test_slice.sum() * first_der_factor) / test_slice.size)

        if abs(first_deri_series[brake]) > test_sum:
            # second derivative criterion:
            slice_start = brake - 12 * pd.Timedelta(data_rate)
            slice_end = brake + 12 * pd.Timedelta(data_rate)
            data_slice = data_slice[slice_start:slice_end]

            # obtain second derivative:
            if smooth is True:
                second_deri_series = pd.Series(
                    data=savgol_filter(
                        data_slice, window_length=smoothing_periods, polyorder=smooth_poly_deg, deriv=2,
                    ),
                    index=data_slice.index,
                )
            else:
                second_deri_series = data_slice.diff().diff()

            # criterion evaluation:
            first_second = (
                (1 - scnd_der_ratio_range)
                < abs((second_deri_series.shift(+1)[brake] / second_deri_series[brake]))
                < 1 + scnd_der_ratio_range
            )

            second_second = abs(second_deri_series[brake] / second_deri_series.shift(-1)[brake]) > scnd_der_ratio_thresh

            if (~first_second) | (~second_second):
                breaks[brake] = False

        else:
            breaks[brake] = False

    flagger = flagger.setFlags(field, breaks, **kwargs)

    return data, flagger
