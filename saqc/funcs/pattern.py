#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import dtw

from saqc.constants import BAD
from saqc.core.register import flagging
from saqc.lib.tools import customRoller


# todo should we mask `reference` even if the func fail if reference has NaNs
@flagging()
def flagPatternByWavelet(
    data,
    field,
    flags,
    reference,
    widths=(1, 2, 4, 8),
    waveform="mexh",
    flag=BAD,
    **kwargs,
):
    """
    Pattern recognition via wavelets.

    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the wavelet algorithm as
        presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged

    Parameters
    ----------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the data column, you want to correct.

    flags : saqc.Flags
        The flags belonging to ``data``.

    reference: str
        The fieldname in ``data`' which holds the pattern.

    widths: tuple of int
        Widths for wavelet decomposition. [1] recommends a dyadic scale.
        Default: (1,2,4,8)

    waveform: str
        Wavelet to be used for decomposition. Default: 'mexh'. See [2] for a list.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.

    flags : saqc.Flags
        The flags belongiong to `data`.

    References
    ----------

    The underlying pattern recognition algorithm using wavelets is documented here:
    [1] Maharaj, E.A. (2002): Pattern Recognition of Time Series using Wavelets. In: Härdle W., Rönz B. (eds) Compstat. Physica, Heidelberg, 978-3-7908-1517-7.

    The documentation of the python package used for the wavelt decomposition can be found here:
    [2] https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families
    """

    dat = data[field]
    ref = data[reference].to_numpy()
    cwtmat_ref, _ = pywt.cwt(ref, widths, waveform)
    wavepower_ref = np.power(cwtmat_ref, 2)
    len_width = len(widths)
    sz = len(ref)

    assert len_width
    assert sz

    def func(x, y):
        return x.sum() / y.sum()

    def pvalue(chunk):
        cwtmat_chunk, _ = pywt.cwt(chunk, widths, waveform)
        wavepower_chunk = np.power(cwtmat_chunk, 2)

        # Permutation test on Powersum of matrix
        for i in range(len_width):
            x = wavepower_ref[i]
            y = wavepower_chunk[i]
            pval = permutation_test(
                x, y, method="approximate", num_rounds=200, func=func, seed=0
            )
            pval = min(pval, 1 - pval)
        return pval  # noqa # existence ensured by assert

    rolling = customRoller(dat, window=sz, min_periods=sz, forward=True)
    pvals = rolling.apply(pvalue, raw=True)
    markers = pvals > 0.01  # nans -> False

    # the markers are set on the left edge of the window. thus we must propagate
    # `sz`-many True's to the right of every marker.
    rolling = customRoller(markers, window=sz, min_periods=sz)
    mask = rolling.sum().fillna(0).astype(bool)

    flags[mask, field] = flag
    return data, flags


def calculateDistanceByDTW(
    data: pd.Series, reference: pd.Series, forward=True, normalize=True
):
    """
    Calculate the DTW-distance of data to pattern in a rolling calculation.

    The data is compared to pattern in a rolling window.
    The size of the rolling window is determined by the timespan defined
    by the first and last timestamp of the reference data's datetime index.

    For details see the linked functions in the `See Also` section.

    Parameters
    ----------
    data : pd.Series
        Data series. Must have datetime-like index, and must be regularly sampled.

    reference : : pd.Series
        Reference series. Must have datetime-like index, must not contain NaNs
        and must not be empty.

    forward: bool, default True
        If `True`, the distance value is set on the left edge of the data chunk. This
        means, with a perfect match, `0.0` marks the beginning of the pattern in
        the data. If `False`, `0.0` would mark the end of the pattern.

    normalize : bool, default True
        If `False`, return unmodified distances.
        If `True`, normalize distances by the number of observations in the reference.
        This helps to make it easier to find a good cutoff threshold for further
        processing. The distances then refer to the mean distance per datapoint,
        expressed in the datas units.

    Returns
    -------
    distance : pd.Series

    Notes
    -----
    The data must be regularly sampled, otherwise a ValueError is raised.
    NaNs in the data will be dropped before dtw distance calculation.

    See Also
    --------
    flagPatternByDTW : flag data by DTW
    """
    if reference.hasnans or reference.empty:
        raise ValueError("reference must not have nan's and must not be empty.")

    winsz = reference.index.max() - reference.index.min()
    reference = reference.to_numpy()

    def isPattern(chunk):
        return dtw.accelerated_dtw(chunk, reference, "euclidean")[0]

    # generate distances, excluding NaNs
    rolling = customRoller(
        data.dropna(), window=winsz, forward=forward, expand=False, closed="both"
    )
    distances: pd.Series = rolling.apply(isPattern, raw=True)

    if normalize:
        distances /= len(reference)

    return distances.reindex(index=data.index)  # reinsert NaNs


# todo should we mask `reference` even if the func fail if reference has NaNs
@flagging()
def flagPatternByDTW(
    data,
    field,
    flags,
    reference,
    max_distance=0.0,
    normalize=True,
    plot=False,
    flag=BAD,
    **kwargs,
):
    """Pattern Recognition via Dynamic Time Warping.

    The steps are:
     1. work on a moving window
     2. for each data chunk extracted from each window, a distance to the given pattern
        is calculated, by the dynamic time warping algorithm [1]
     3. if the distance is below the threshold, all the data in the window gets flagged

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The name of the data column

    flags : saqc.Flags
        The flags belonging to `data`.

    reference : str
        The name in `data` which holds the pattern. The pattern must not have NaNs,
        have a datetime index and must not be empty.

    max_distance : float, default 0.0
        Maximum dtw-distance between chunk and pattern, if the distance is lower than
        ``max_distance`` the data gets flagged. With default, ``0.0``, only exact
        matches are flagged.

    normalize : bool, default True
        If `False`, return unmodified distances.
        If `True`, normalize distances by the number of observations of the reference.
        This helps to make it easier to find a good cutoff threshold for further
        processing. The distances then refer to the mean distance per datapoint,
        expressed in the datas units.

    plot: bool, default False
        Show a calibration plot, which can be quite helpful to find the right threshold
        for `max_distance`. It works best with `normalize=True`. Do not use in automatic
        setups / pipelines. The plot show three lines:

        - data: the data the function was called on
        - distances: the calculated distances by the algorithm
        - indicator: have to distinct levels: `0` and the value of `max_distance`.
          If `max_distance` is `0.0` it defaults to `1`. Everywhere where the
          indicator is not `0` the data will be flagged.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.

    flags : saqc.Flags
        The flags belonging to `data`.

    Notes
    -----
    The window size of the moving window is set to equal the temporal extension of the
    reference datas datetime index.

    References
    ----------
    Find a nice description of underlying the Dynamic Time Warping Algorithm here:

    [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
    """
    ref = data[reference]
    dat = data[field]

    distances = calculateDistanceByDTW(dat, ref, forward=True, normalize=normalize)
    winsz = ref.index.max() - ref.index.min()

    # prevent nan propagation
    distances = distances.fillna(max_distance + 1)

    # find minima filter by threshold
    fw = customRoller(distances, window=winsz, forward=True, closed="both", expand=True)
    bw = customRoller(distances, window=winsz, closed="both", expand=True)
    minima = (fw.min() == bw.min()) & (distances <= max_distance)

    # Propagate True's to size of pattern.
    rolling = customRoller(minima, window=winsz, closed="both", expand=True)
    mask = rolling.sum() > 0

    if plot:
        df = pd.DataFrame()
        df["data"] = dat
        df["distances"] = distances
        df["indicator"] = mask.astype(float) * (max_distance or 1)
        df.plot()

    flags[mask, field] = flag
    return data, flags
