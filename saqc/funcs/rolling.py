#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Callable

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.core.register import register
from saqc.flagger import Flagger
from saqc.lib.tools import getFreqDelta


@register(masking='field', module="rolling")
def roll(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        winsz: Union[str, int],
        func: Callable[[pd.Series], float]=np.mean,
        eval_flags: bool=True,
        min_periods: int=0,
        center: bool=True,
        return_residues=False,
        **kwargs
):
    """
        Models the data with the rolling mean and returns the residues.

        Note, that the residues will be stored to the `field` field of the input data, so that the data that is modelled
        gets overridden.

        Parameters
        ----------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
        field : str
            The fieldname of the column, holding the data-to-be-modelled.
        flagger : saqc.flagger.Flagger
            A flagger object, holding flags and additional Informations related to `data`.
        winsz : {int, str}
            The size of the window you want to roll with. If an integer is passed, the size
            refers to the number of periods for every fitting window. If an offset string is passed,
            the size refers to the total temporal extension.
            For regularly sampled timeseries, the period number will be casted down to an odd number if
            center = True.
        func : Callable[np.array, float], default np.mean
            Function to apply on the rolling window and obtain the curve fit value.
        eval_flags : bool, default True
            Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
            flag present in the interval, the data for its calculation was obtained from.
            Currently not implemented in combination with not-harmonized timeseries.
        min_periods : int, default 0
            The minimum number of periods, that has to be available in every values fitting surrounding for the mean
            fitting to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
            regardless of the number of values present.
        center : bool, default True
            Wheather or not to center the window the mean is calculated of around the reference value. If False,
            the reference value is placed to the right of the window (classic rolling mean with lag.)

        Returns
        -------
        data : dios.DictOfSeries
            A dictionary of pandas.Series, holding all the data.
            Data values may have changed relatively to the data input.
        flagger : saqc.flagger.Flagger
            The flagger object, holding flags and additional Informations related to `data`.
            Flags values may have changed relatively to the flagger input.
        """

    data = data.copy()
    to_fit = data[field]
    if to_fit.empty:
        return data, flagger

    regular = getFreqDelta(to_fit.index)
    # starting with the annoying case: finding the rolling interval centers of not-harmonized input time series:
    if center and not regular:
        if isinstance(winsz, int):
            raise NotImplementedError(
                "Integer based window size is not supported for not-harmonized"
                'sample series when rolling with "center=True".'
            )
        # get interval centers
        centers = np.floor((to_fit.rolling(pd.Timedelta(winsz) / 2, closed="both", min_periods=min_periods).count()))
        centers = centers.drop(centers[centers.isna()].index)
        centers = centers.astype(int)
        roller = to_fit.rolling(pd.Timedelta(winsz), closed="both", min_periods=min_periods)
        try:
            means = getattr(roller, func.__name__)()
        except AttributeError:
            means = to_fit.rolling(pd.Timedelta(winsz), closed="both", min_periods=min_periods).apply(func)

        def center_func(x, y=centers):
            pos = x.index[int(len(x) - y[x.index[-1]])]
            return y.index.get_loc(pos)

        centers_iloc = centers.rolling(winsz, closed="both").apply(center_func, raw=False).astype(int)
        temp = means.copy()
        for k in centers_iloc.iteritems():
            means.iloc[k[1]] = temp[k[0]]
        # last values are false, due to structural reasons:
        means[means.index[centers_iloc[-1]]: means.index[-1]] = np.nan

    # everything is more easy if data[field] is harmonized:
    else:
        if isinstance(winsz, str):
            winsz = pd.Timedelta(winsz) // regular
        if (winsz % 2 == 0) & center:
            winsz = int(winsz - 1)

        roller = to_fit.rolling(window=winsz, center=center, closed="both")
        try:
            means = getattr(roller, func.__name__)()
        except AttributeError:
            means = to_fit.rolling(window=winsz, center=center, closed="both").apply(func)

    if return_residues:
        means = to_fit - means

    data[field] = means
    if eval_flags:
        # with the new flagger we dont have to care
        # about to set NaNs to the original flags anymore
        # todo: we does not get any flags here, because of masking=field
        worst = flagger[field].rolling(winsz, center=True, min_periods=min_periods).max()
        flagger[field] = worst

    return data, flagger
