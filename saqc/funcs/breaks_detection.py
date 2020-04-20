#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from saqc.funcs.register import register
from saqc.lib.tools import retrieveTrustworthyOriginal


@register()
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

    """ This Function is an generalization of the Spectrum based break flagging mechanism as presented in:

    Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture Data from the international
    Soil Moisture Network. 2013. Vadoze Zone J. doi:10.2136/vzj2012.0097.

    The function flags breaks (jumps/drops) in input measurement series by evaluating its derivatives.
    A measurement y_t is flagged a, break, if:

    (1) y_t is changing relatively to its preceeding value by at least (100*rel_change_rate_min) percent
    (2) y_(t-1) is difffering from its preceeding value, by a margin of at least "thresh_abs"
    (3) Absolute first derivative |(y_t)'| has to be at least "first_der_factor" times as big as the arithmetic middle
        over all the first derivative values within a 2 times "first_der_window_size" hours window, centered at t.
    (4) The ratio of the second derivatives at t and t+1 has to be "aproximately" 1.
        ([1-scnd__der_ration_margin_1, 1+scnd_ratio_margin_1])
    (5) The ratio of the second derivatives at t+1 and t+2 has to be larger than scnd_der_ratio_margin_2

    NOTE 1: As no reliable statement about the plausibility of the meassurements before and after the jump is possible,
    only the jump itself is flagged. For flagging constant values following upon a jump, use a flagConstants test.

    NOTE 2: All derivatives in the reference publication are obtained by applying a Savitzky-Golay filter to the data
    before differentiating. However, i was not able to reproduce satisfaction of all the conditions for synthetically
    constructed breaks.
    Especially condition [4] and [5]! This is because smoothing distributes the harshness of the break over the
    smoothing window. Since just taking the differences as derivatives did work well for my empirical data set,
    the parameter "smooth" defaults to "raw". That means, that derivatives will be obtained by just using the
    differences series.
    You are free of course, to change this parameter to "savgol" and play around with the associated filter options.
    (see parameter description below)




       :param data:                        The pandas dataframe holding the data-to-be flagged.
                                           Data must be indexed by a datetime series and be harmonized onto a
                                           time raster with seconds precision (skips allowed).
       :param flags:                       A dataframe holding the flags/flag-entries associated with "data".
       :param field:                       Fieldname of the Soil moisture measurements field in data.
       :param flagger:                     A flagger - object. (saqc.flagger.X)
       :param smooth:                      Bool. Method for obtaining dataseries' derivatives.
                                           False: Just take series step differences (default)
                                           True: Smooth data with a Savitzky Golay Filter before differentiating.
       :param smooth_window:               Offset string. Size of the filter window, used to calculate the derivatives.
                                           (relevant only, if: smooth is True)
       :param smooth_poly_deg:             Integer. Polynomial order, used for smoothing with savitzk golay filter.
                                           (relevant only, if: smooth_func='savgol')
       :param thresh_rel                   Float in [0,1]. See (1) of function descritpion above to learn more
       :param thresh_abs                   Float > 0. See (2) of function descritpion above to learn more.
       :param first_der_factor             Float > 0. See (3) of function descritpion above to learn more.
       :param first_der_window_range        Offset_String. See (3) of function description to learn more.
       :param scnd_der_ratio_margin_1      Float in [0,1]. See (4) of function descritpion above to learn more.
       :param scnd_der_ratio_margin_2      Float in [0,1]. See (5) of function descritpion above to learn more.
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
    breaks = breaks[breaks == True]

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

    breaks = breaks[breaks == True]

    flagger = flagger.setFlags(field, breaks.index, **kwargs)

    return data, flagger
