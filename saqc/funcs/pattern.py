#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Union, Tuple, Optional
import numpy as np
import dtw
import pywt
from mlxtend.evaluate import permutation_test
from dios.dios import DictOfSeries

from saqc.core.register import register
from saqc.flagger import Flagger
from saqc.lib.tools import customRoller


@register(masking='field', module="pattern")
def flagPatternByDTW(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        ref_field: str,
        widths: Sequence[int]=(1, 2, 4, 8),
        waveform: str="mexh",
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Pattern recognition via wavelets.

    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the wavelet algorithm as presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged

    Parameters
    ----------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    ref_field: str
        The fieldname in `data' which holds the pattern.
    widths: tuple of int
        Widths for wavelet decomposition. [1] recommends a dyadic scale. Default: (1,2,4,8)
    waveform: str.
        Wavelet to be used for decomposition. Default: 'mexh'. See [2] for a list.

    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.


    References
    ----------

    The underlying pattern recognition algorithm using wavelets is documented here:
    [1] Maharaj, E.A. (2002): Pattern Recognition of Time Series using Wavelets. In: Härdle W., Rönz B. (eds) Compstat. Physica, Heidelberg, 978-3-7908-1517-7.

    The documentation of the python package used for the wavelt decomposition can be found here:
    [2] https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families
    """

    ref = data[ref_field].to_numpy()
    cwtmat_ref, _ = pywt.cwt(ref, widths, waveform)
    wavepower_ref = np.power(cwtmat_ref, 2)
    len_width = len(widths)

    def func(x, y):
        return x.sum() / y.sum()

    def isPattern(chunk):
        cwtmat_chunk, _ = pywt.cwt(chunk, widths, waveform)
        wavepower_chunk = np.power(cwtmat_chunk, 2)

        # Permutation test on Powersum of matrix
        for i in range(len_width):
            x = wavepower_ref[i]
            y = wavepower_chunk[i]
            pval = permutation_test(x, y, method='approximate', num_rounds=200, func=func, seed=0)
            if min(pval, 1 - pval) > 0.01:
                return True
        return False

    dat = data[field]
    sz = len(ref)
    mask = customRoller(dat, window=sz, min_periods=sz).apply(isPattern, raw=True)

    flagger[mask, field] = kwargs['flag']
    return data, flagger


@register(masking='field', module="pattern")
def flagPatternByWavelet(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        ref_field: str,
        max_distance: float=0.03,
        normalize: bool=True,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """ Pattern Recognition via Dynamic Time Warping.

    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the dynamic time warping algorithm as presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged

    Parameters
    ----------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    ref_field: str
        The fieldname in `data` which holds the pattern.
    max_distance: float
        Maximum dtw-distance between partition and pattern, so that partition is recognized as pattern. Default: 0.03
    normalize: boolean.
        Normalizing dtw-distance (see [1]). Default: True


    kwargs

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.


    References
    ----------
    Find a nice description of underlying the Dynamic Time Warping Algorithm here:

    [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
    """
    ref = data[ref_field]
    ref_var = ref.var()

    def func(a, b):
        return np.linalg.norm(a - b)

    def isPattern(chunk):
        dist, *_ = dtw.dtw(chunk, ref, func)
        if normalize:
            dist /= ref_var
        return dist < max_distance

    dat = data[field]
    sz = len(ref)
    mask = customRoller(dat, window=sz, min_periods=sz).apply(isPattern, raw=True)

    flagger[mask, field] = kwargs['flag']
    return data, flagger
