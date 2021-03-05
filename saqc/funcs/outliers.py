#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, Tuple, Sequence, Callable
from typing_extensions import Literal

import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
import pandas as pd
import numba

from outliers import smirnov_grubbs

from dios import DictOfSeries

from saqc.common import *
from saqc.core.register import register
from saqc.flagger import Flagger
from saqc.lib.tools import (
    customRoller,
    findIndex,
    getFreqDelta
)
from saqc.lib.types import ColumnName, FreqString, IntegerWindow
from saqc.funcs.scores import assignKNNScore
import saqc.lib.ts_operators as ts_ops


@register(masking='field', module="outliers")
def flagByStray(
    data: DictOfSeries,
    field: ColumnName,
    flagger: Flagger,
    partition_freq: Optional[Union[IntegerWindow, FreqString]]=None,
    partition_min: int=11,
    iter_start: float=0.5,
    alpha: float=0.05,
    **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Flag outliers in 1-dimensional (score) data with the STRAY Algorithm.

    Find more information on the algorithm in References [1].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.

    partition_freq : str, int, or None, default None
        Determines the segmentation of the data into partitions, the kNN algorithm is
        applied onto individually.

        * ``np.inf``: Apply Scoring on whole data set at once
        * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
        * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
          string

    partition_min : int, default 11
        Minimum number of periods per partition that have to be present for a valid outlier dettection to be made in
        this partition. (Only of effect, if `partition_freq` is an integer.) Partition min value must always be
        greater then the nn_neighbors value.

    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered "normal". 0.5 results in the stray
        algorithm to search only the upper 50 % of the scores for the cut off point. (See reference section for more
        information)

    alpha : float, default 0.05
        Level of significance by which it is tested, if a score might be drawn from another distribution, than the
        majority of the data.

    References
    ----------
    [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in high dimensional data.
        arXiv preprint arXiv:1908.04000.
    """
    scores = data[field].dropna()

    if scores.empty:
        return data, flagger

    if not partition_freq:
        partition_freq = scores.shape[0]

    if isinstance(partition_freq, str):
        partitions = scores.groupby(pd.Grouper(freq=partition_freq))

    else:
        grouper_series = pd.Series(data=np.arange(0, scores.shape[0]), index=scores.index)
        grouper_series = grouper_series.transform(lambda x: int(np.floor(x / partition_freq)))
        partitions = scores.groupby(grouper_series)

    # calculate flags for every partition
    for _, partition in partitions:

        if partition.empty | (partition.shape[0] < partition_min):
            continue

        sample_size = partition.shape[0]

        sorted_i = partition.values.argsort()
        resids = partition.values[sorted_i]
        gaps = np.append(0, np.diff(resids))

        tail_size = int(max(min(50, np.floor(sample_size / 4)), 2))
        tail_indices = np.arange(2, tail_size + 1)

        i_start = int(max(np.floor(sample_size * iter_start), 1) + 1)
        ghat = np.array([np.nan] * sample_size)

        for i in range(i_start - 1, sample_size):
            ghat[i] = sum((tail_indices / (tail_size - 1)) * gaps[i - tail_indices + 1])

        log_alpha = np.log(1 / alpha)
        for iter_index in range(i_start - 1, sample_size):
            if gaps[iter_index] > log_alpha * ghat[iter_index]:
                index = partition.index[sorted_i[iter_index:]]
                flagger[index, field] = kwargs['flag']
                break

    return data, flagger


def _evalStrayLabels(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        fields: Sequence[str],
        reduction_range: Optional[str]=None,
        reduction_drop_flagged: bool=False,
        reduction_thresh: float=3.5,
        reduction_min_periods: int=1,
        at_least_one: bool=True,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function "reduces" an observations flag to components of it, by applying MAD (See references)
    test onto every components temporal surrounding.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the labels to be evaluated.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    fields : list[str]
        A list of strings, holding the column names of the variables, the stray labels shall be
        projected onto.
    val_frame : (N,M) pd.DataFrame
        Input NxM DataFrame of observations, where N is the number of observations and M the number of components per
        observation.
    to_flag_frame : pandas.DataFrame
        Input dataframe of observations to be tested, where N is the number of observations and M the number
        of components per observation.
    reduction_range : {None, str}
        An offset string, denoting the range of the temporal surrounding to include into the MAD testing.
        If ``None`` is passed, no testing will be performed and all fields will have the stray flag projected.
    reduction_drop_flagged : bool, default False
        Wheather or not to drop flagged values other than the value under test, from the temporal surrounding
        before checking the value with MAD.
    reduction_thresh : float, default 3.5
        The `critical` value, controlling wheather the MAD score is considered referring to an outlier or not.
        Higher values result in less rigid flagging. The default value is widely used in the literature. See references
        section for more details ([1]).
    at_least_one : bool, default True
        If none of the variables, the outlier label shall be reduced to, is an outlier with regard
        to the test, all (True) or none (False) of the variables are flagged

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    val_frame = data[fields].to_df()
    stray_detects = flagger[field] > UNFLAGGED
    stray_detects = stray_detects[stray_detects]
    to_flag_frame = pd.DataFrame(False, columns=fields, index=stray_detects.index)

    if reduction_range is None:
        for field in to_flag_frame.columns:
            flagger[to_flag_frame.index, field] = kwargs['flag']
        return data, flagger

    for var in fields:
        for index in enumerate(to_flag_frame.index):

            index_slice = slice(index[1] - pd.Timedelta(reduction_range), index[1] + pd.Timedelta(reduction_range))
            test_slice = val_frame[var][index_slice].dropna()

            # check, wheather value under test is sufficiently centered:
            first = test_slice.first_valid_index()
            last = test_slice.last_valid_index()
            min_range = pd.Timedelta(reduction_range)/4

            if pd.Timedelta(index[1] - first) < min_range or pd.Timedelta(last - index[1]) < min_range:
                polydeg = 0
            else:
                polydeg = 2

            if reduction_drop_flagged:
                test_slice = test_slice.drop(to_flag_frame.index, errors='ignore')

            if test_slice.shape[0] < reduction_min_periods:
                to_flag_frame.loc[index[1], var] = True
                continue

            x = (test_slice.index.values.astype(float))
            x_0 = x[0]
            x = (x - x_0)/10**12

            polyfitted = poly.polyfit(y=test_slice.values, x=x, deg=polydeg)

            testval = poly.polyval((float(index[1].to_numpy()) - x_0)/10**12, polyfitted)
            testval = val_frame[var][index[1]] - testval

            resids = test_slice.values - poly.polyval(x, polyfitted)
            med_resids = np.median(resids)
            MAD = np.median(np.abs(resids - med_resids))
            crit_val = 0.6745 * (abs(med_resids - testval)) / MAD

            if crit_val > reduction_thresh:
                to_flag_frame.loc[index[1], var] = True

    if at_least_one:
        to_flag_frame[~to_flag_frame.any(axis=1)] = True

    for field in to_flag_frame.columns:
        col = to_flag_frame[field]
        flagger[col[col].index, field] = kwargs['flag']

    return data, flagger


def _expFit(val_frame, scoring_method="kNNMaxGap", n_neighbors=10, iter_start=0.5, alpha=0.05, bin_frac=10):
    """
    Find outliers in multi dimensional observations.

    The general idea is to assigning scores to every observation based on the observations neighborhood in the space
    of observations. Then, the gaps between the (greatest) scores are tested for beeing drawn from the same
    distribution, as the majority of the scores.

    Note, that no normalizations/transformations are applied to the different components (data columns)
    - those are expected to be applied previously, if necessary.

    Parameters
    ----------
    val_frame : (N,M) ndarray
        Input NxM array of observations, where N is the number of observations and M the number of components per
        observation.
    scoring_method : {'kNNSum', 'kNNMaxGap'}, default 'kNNMaxGap'
        Scoring method applied.
        `'kNNSum'`: Assign to every point the sum of the distances to its 'n_neighbors' nearest neighbors.
        `'kNNMaxGap'`: Assign to every point the distance to the neighbor with the "maximum gap" to its predecessor
        in the hierarchy of the `n_neighbors` nearest neighbors. (see reference section for further descriptions)
    n_neighbors : int, default 10
        Number of neighbors included in the scoring process for every datapoint.
    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered "normal". 0.5 results in the expfit
        algorithm to search only the upper 50 % of the scores for the cut off point. (See reference section for more
        information)
    alpha : float, default 0.05
        Niveau of significance by which it is tested, if a score might be drawn from another distribution, than the
        majority of the data.
    bin_frac : {int, str}, default 10
        Controls the binning for the histogram in the fitting step. If an integer is passed, the residues will
        equidistantly be covered by `bin_frac` bins, ranging from the minimum to the maximum of the residues.
        If a string is passed, it will be passed on to the ``numpy.histogram_bin_edges`` method.
    """

    kNNfunc = getattr(ts_ops, scoring_method)
    resids = kNNfunc(val_frame.values, n_neighbors=n_neighbors, algorithm="ball_tree")
    data_len = resids.shape[0]

    # sorting
    sorted_i = resids.argsort()
    resids = resids[sorted_i]
    iter_index = int(np.floor(resids.size * iter_start))
    # initialize condition variables:
    crit_val = np.inf
    test_val = 0
    neg_log_alpha = -np.log(alpha)

    # define exponential dist density function:
    def fit_function(x, lambd):
        return lambd * np.exp(-lambd * x)

    # initialise sampling bins
    if isinstance(bin_frac, int):
        binz = np.linspace(resids[0], resids[-1], 10 * int(np.ceil(data_len / bin_frac)))
    elif bin_frac in ["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]:
        binz = np.histogram_bin_edges(resids, bins=bin_frac)
    else:
        raise ValueError(f"Can't interpret {bin_frac} as an binning technique.")

    binzenters = np.array([0.5 * (binz[i] + binz[i + 1]) for i in range(len(binz) - 1)])
    # inititialize full histogram:
    full_hist, binz = np.histogram(resids, bins=binz)
    # check if start index is sufficiently high (pointing at resids value beyond histogram maximum at least):
    hist_argmax = full_hist.argmax()

    if hist_argmax >= findIndex(binz, resids[iter_index - 1], 0):
        raise ValueError(
            "Either the data histogram is too strangely shaped for oddWater OD detection - "
            "or a too low value for 'iter_start' was passed "
            "(iter_start better be much greater 0.5)"
        )
    # GO!
    iter_max_bin_index = findIndex(binz, resids[iter_index - 1], 0)
    upper_tail_index = int(np.floor(0.5 * hist_argmax + 0.5 * iter_max_bin_index))
    resids_tail_index = findIndex(resids, binz[upper_tail_index], 0)
    upper_tail_hist, bins = np.histogram(
        resids[resids_tail_index:iter_index], bins=binz[upper_tail_index : iter_max_bin_index + 1]
    )

    while (test_val < crit_val) & (iter_index < resids.size - 1):
        iter_index += 1
        new_iter_max_bin_index = findIndex(binz, resids[iter_index - 1], 0)
        # following if/else block "manually" expands the data histogram and circumvents calculation of the complete
        # histogram in any new iteration.
        if new_iter_max_bin_index == iter_max_bin_index:
            upper_tail_hist[-1] += 1
        else:
            upper_tail_hist = np.append(upper_tail_hist, np.zeros([new_iter_max_bin_index - iter_max_bin_index]))
            upper_tail_hist[-1] += 1
            iter_max_bin_index = new_iter_max_bin_index
            upper_tail_index_new = int(np.floor(0.5 * hist_argmax + 0.5 * iter_max_bin_index))
            upper_tail_hist = upper_tail_hist[upper_tail_index_new - upper_tail_index :]
            upper_tail_index = upper_tail_index_new

        # fitting

        lambdA, _ = curve_fit(
            fit_function,
            xdata=binzenters[upper_tail_index:iter_max_bin_index],
            ydata=upper_tail_hist,
            p0=[-np.log(alpha / resids[iter_index])],
        )

        crit_val = neg_log_alpha / lambdA
        test_val = resids[iter_index]

    return val_frame.index[sorted_i[iter_index:]]


@register(masking='all', module="outliers")
def flagMVScores(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        fields: Sequence[ColumnName],
        trafo: Callable[[pd.Series], pd.Series]=lambda x: x,
        alpha: float=0.05,
        n_neighbors: int=10,
        scoring_func: Callable[[pd.Series], float]=np.sum,
        iter_start: float=0.5,
        stray_partition: Optional[Union[IntegerWindow, FreqString]]=None,
        stray_partition_min: int=11,
        trafo_on_partition: bool=True,
        reduction_range: Optional[FreqString]=None,
        reduction_drop_flagged: bool=False,
        reduction_thresh: float=3.5,
        reduction_min_periods: int=1,
        **kwargs,
) -> Tuple[DictOfSeries, Flagger]:
    """
    The algorithm implements a 3-step outlier detection procedure for simultaneously flagging of higher dimensional
    data (dimensions > 3).

    In references [1], the procedure is introduced and exemplified with an application on hydrological data.

    See the notes section for an overview over the algorithms basic steps.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    fields : List[str]
        List of fieldnames, corresponding to the variables that are to be included into the flagging process.
    trafo : callable, default lambda x:x
        Transformation to be applied onto every column before scoring. Will likely get deprecated soon. Its better
        to transform the data in a processing step, preceeeding the call to ``flagMVScores``.
    alpha : float, default 0.05
        Level of significance by which it is tested, if an observations score might be drawn from another distribution
        than the majority of the observation.
    n_neighbors : int, default 10
        Number of neighbors included in the scoring process for every datapoint.
    scoring_func : Callable[numpy.array, float], default np.sum
        The function that maps the set of every points k-nearest neighbor distances onto a certain scoring.
    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered "normal". 0.5 results in the threshing
        algorithm to search only the upper 50 % of the scores for the cut off point. (See reference section for more
        information)
    stray_partition : {None, str, int}, default None
        Only effective when `threshing` = 'stray'.
        Determines the size of the data partitions, the data is decomposed into. Each partition is checked seperately
        for outliers. If a String is passed, it has to be an offset string and it results in partitioning the data into
        parts of according temporal length. If an integer is passed, the data is simply split up into continous chunks
        of `partition_freq` periods. if ``None`` is passed (default), all the data will be tested in one run.
    stray_partition_min : int, default 11
        Only effective when `threshing` = 'stray'.
        Minimum number of periods per partition that have to be present for a valid outlier detection to be made in
        this partition. (Only of effect, if `stray_partition` is an integer.)
    trafo_on_partition : bool, default True
        Whether or not to apply the passed transformation on every partition the algorithm is applied on, separately.
    reduction_range : {None, str}, default None
        If not None, it is tried to reduce the stray result onto single outlier components of the input fields.
        An offset string, denoting the range of the temporal surrounding to include into the MAD testing while trying
        to reduce flags.
    reduction_drop_flagged : bool, default False
        Only effective when `reduction_range` is not ``None``.
        Whether or not to drop flagged values other than the value under test from the temporal surrounding
        before checking the value with MAD.
    reduction_thresh : float, default 3.5
        Only effective when `reduction_range` is not ``None``.
        The `critical` value, controlling wheather the MAD score is considered referring to an outlier or not.
        Higher values result in less rigid flagging. The default value is widely considered apropriate in the
        literature.
    reduction_min_periods : int, 1
        Only effective when `reduction_range` is not ``None``.
        Minimum number of meassurements necessarily present in a reduction interval for reduction actually to be
        performed.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    Notes
    -----
    The basic steps are:

    1. transforming

    The different data columns are transformed via timeseries transformations to
    (a) make them comparable and
    (b) make outliers more stand out.

    This step is usually subject to a phase of research/try and error. See [1] for more details.

    Note, that the data transformation as an built-in step of the algorithm, will likely get deprecated soon. Its better
    to transform the data in a processing step, preceeding the multivariate flagging process. Also, by doing so, one
    gets mutch more control and variety in the transformation applied, since the `trafo` parameter only allows for
    application of the same transformation to all of the variables involved.

    2. scoring

    Every observation gets assigned a score depending on its k nearest neighbors. See the `scoring_method` parameter
    description for details on the different scoring methods. Furthermore [1], [2] may give some insight in the
    pro and cons of the different methods.

    3. threshing

    The gaps between the (greatest) scores are tested for beeing drawn from the same
    distribution as the majority of the scores. If a gap is encountered, that, with sufficient significance, can be
    said to not be drawn from the same distribution as the one all the smaller gaps are drawn from, than
    the observation belonging to this gap, and all the observations belonging to gaps larger then this gap, get flagged
    outliers. See description of the `threshing` parameter for more details. Although [2] gives a fully detailed
    overview over the `stray` algorithm.
    """
    data, flagger = assignKNNScore(
        data, 'dummy', flagger,
        fields=fields,
        n_neighbors=n_neighbors,
        trafo=trafo,
        trafo_on_partition=trafo_on_partition,
        scoring_func=scoring_func,
        target_field='kNN_scores',
        partition_freq=stray_partition,
        kNN_algorithm='ball_tree',
        partition_min=stray_partition_min, **kwargs)

    data, flagger = flagByStray(
        data, 'kNN_scores', flagger,
        partition_freq=stray_partition,
        partition_min=stray_partition_min,
        iter_start=iter_start,
        alpha=alpha, **kwargs)

    data, flagger = _evalStrayLabels(
        data, 'kNN_scores', flagger,
        fields=fields,
        reduction_range=reduction_range,
        reduction_drop_flagged=reduction_drop_flagged,
        reduction_thresh=reduction_thresh,
        reduction_min_periods=reduction_min_periods, **kwargs)

    return data, flagger


@register(masking='field', module="outliers")
def flagRaise(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        thresh: float,
        raise_window: FreqString,
        intended_freq: FreqString,
        average_window: Optional[FreqString]=None,
        mean_raise_factor: float=2.,
        min_slope: Optional[float]=None,
        min_slope_weight: float=0.8,
        numba_boost: bool=True,  # TODO: rm, not a user decision
        **kwargs,
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function flags raises and drops in value courses, that exceed a certain threshold
    within a certain timespan.

    The parameter variety of the function is owned to the intriguing
    case of values, that "return" from outlierish or anomalious value levels and
    thus exceed the threshold, while actually being usual values.

    NOTE, the dataset is NOT supposed to be harmonized to a time series with an
    equidistant frequency grid.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    thresh : float
        The threshold, for the total rise (thresh > 0), or total drop (thresh < 0), value courses must
        not exceed within a timespan of length `raise_window`.
    raise_window : str
        An offset string, determining the timespan, the rise/drop thresholding refers to. Window is inclusively defined.
    intended_freq : str
        An offset string, determining The frequency, the timeseries to-be-flagged is supposed to be sampled at.
        The window is inclusively defined.
    average_window : {None, str}, default None
        See condition (2) of the description linked in the references. Window is inclusively defined.
        The window defaults to 1.5 times the size of `raise_window`
    mean_raise_factor : float, default 2
        See second condition listed in the notes below.
    min_slope : {None, float}, default None
        See third condition listed in the notes below.
    min_slope_weight : float, default 0.8
        See third condition listed in the notes below.
    numba_boost : bool, default True

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    Notes
    -----
    The value :math:`x_{k}` of a time series :math:`x` with associated
    timestamps :math:`t_i`, is flagged a raise, if:

    * There is any value :math:`x_{s}`, preceeding :math:`x_{k}` within `raise_window` range, so that:

      * :math:`M = |x_k - x_s | >`  `thresh` :math:`> 0`

    * The weighted average :math:`\\mu^{*}` of the values, preceding :math:`x_{k}` within `average_window`
      range indicates, that :math:`x_{k}` does not return from an "outlierish" value course, meaning that:

      * :math:`x_k > \\mu^* + ( M` / `mean_raise_factor` :math:`)`

    * Additionally, if `min_slope` is not `None`, :math:`x_{k}` is checked for being sufficiently divergent from its
      very predecessor :max:`x_{k-1}`$, meaning that, it is additionally checked if:

      * :math:`x_k - x_{k-1} >` `min_slope`
      * :math:`t_k - t_{k-1} >` `min_slope_weight` :math:`\\times` `intended_freq`

    """

    # prepare input args
    dataseries = data[field].dropna()
    raise_window = pd.Timedelta(raise_window)
    intended_freq = pd.Timedelta(intended_freq)
    if min_slope is not None:
        min_slope = np.abs(min_slope)

    if average_window is None:
        average_window = 1.5 * pd.Timedelta(raise_window)

    if thresh < 0:
        dataseries *= -1
        thresh *= -1

    def raise_check(x, thresh):
        test_set = x[-1] - x[0:-1]
        max_val = np.max(test_set)
        if max_val >= thresh:
            return max_val
        else:
            return np.nan

    def custom_rolling_mean(x):
        return np.sum(x[:-1])

    # get invalid-raise/drop mask:
    raise_series = dataseries.rolling(raise_window, min_periods=2, closed="both")

    if numba_boost:
        raise_check = numba.jit(raise_check, nopython=True)
        raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True, engine="numba")
    else:
        raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True)

    if raise_series.isna().all():
        return data, flagger

    # "unflag" values of insufficient deviation to their predecessors
    if min_slope is not None:
        w_mask = (
            pd.Series(dataseries.index).diff().dt.total_seconds() / intended_freq.total_seconds()
        ) > min_slope_weight
        slope_mask = np.abs(dataseries.diff()) < min_slope
        to_unflag = raise_series.notna() & w_mask.values & slope_mask
        raise_series[to_unflag] = np.nan

    # calculate and apply the weighted mean weights (pseudo-harmonization):
    weights = (
        pd.Series(dataseries.index).diff(periods=2).shift(-1).dt.total_seconds() / intended_freq.total_seconds() / 2
    )

    weights.iloc[0] = 0.5 + (dataseries.index[1] - dataseries.index[0]).total_seconds() / (
        intended_freq.total_seconds() * 2
    )

    weights.iloc[-1] = 0.5 + (dataseries.index[-1] - dataseries.index[-2]).total_seconds() / (
        intended_freq.total_seconds() * 2
    )

    weights[weights > 1.5] = 1.5
    weights.index = dataseries.index
    weighted_data = dataseries.mul(weights)

    # rolling weighted mean calculation
    weighted_rolling_mean = weighted_data.rolling(average_window, min_periods=2, closed="both")
    weights_rolling_sum = weights.rolling(average_window, min_periods=2, closed="both")
    if numba_boost:
        custom_rolling_mean = numba.jit(custom_rolling_mean, nopython=True)
        weighted_rolling_mean = weighted_rolling_mean.apply(custom_rolling_mean, raw=True, engine="numba")
        weights_rolling_sum = weights_rolling_sum.apply(custom_rolling_mean, raw=True, engine="numba")
    else:
        weighted_rolling_mean = weighted_rolling_mean.apply(custom_rolling_mean, raw=True)
        weights_rolling_sum = weights_rolling_sum.apply(custom_rolling_mean, raw=True, engine="numba")

    weighted_rolling_mean = weighted_rolling_mean / weights_rolling_sum
    # check means against critical raise value:
    to_flag = dataseries >= weighted_rolling_mean + (raise_series / mean_raise_factor)
    to_flag &= raise_series.notna()
    flagger[to_flag[to_flag].index, field] = kwargs['flag']

    return data, flagger


@register(masking='field', module="outliers")
def flagMAD(
        data: DictOfSeries, field: ColumnName, flagger: Flagger, window: FreqString, z: float=3.5, **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function represents an implementation of the modyfied Z-score outlier detection method.

    See references [1] for more details on the algorithm.

    Note, that the test needs the input data to be sampled regularly (fixed sampling rate).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    window : str
       Offset string. Denoting the windows size that the "Z-scored" values have to lie in.
    z: float, default 3.5
        The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    d = data[field]
    median = d.rolling(window=window, closed="both").median()
    diff = (d - median).abs()
    mad = diff.rolling(window=window, closed="both").median()
    mask = (mad > 0) & (0.6745 * diff > z * mad)
    # NOTE:
    # In pandas <= 0.25.3, the window size is not fixed if the
    # window-argument to rolling is a frequency. That implies,
    # that during the first iterations the window has a size of
    # 1, 2, 3, ... until it eventually covers the disered time
    # span. For stuff the calculation of median, that is rather
    # unfortunate, as the size of calculation base might differ
    # heavily. So don't flag something until, the window reaches
    # its target size
    if not isinstance(window, int):
        index = mask.index
        mask.loc[index < index[0] + pd.to_timedelta(window)] = False

    flagger[mask, field] = kwargs['flag']
    return data, flagger


@register(masking='field', module="outliers")
def flagOffset(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        thresh: float,
        tolerance: float,
        window: Union[IntegerWindow, FreqString],
        rel_thresh: Optional[float]=None,
        numba_kickin: int=200000,  # TODO: rm, not a user decision
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    A basic outlier test that is designed to work for harmonized and not harmonized data.

    The test classifies values/value courses as outliers by detecting not only a rise in value, but also,
    checking for a return to the initial value level.

    Values :math:`x_n, x_{n+1}, .... , x_{n+k}` of a timeseries :math:`x` with associated timestamps
    :math:`t_n, t_{n+1}, .... , t_{n+k}` are considered spikes, if

    1. :math:`|x_{n-1} - x_{n + s}| >` `thresh`, for all :math:`s \\in [0,1,2,...,k]`

    2. :math:`|x_{n-1} - x_{n+k+1}| <` `tolerance`

    3. :math:`|t_{n-1} - t_{n+k+1}| <` `window`

    Note, that this definition of a "spike" not only includes one-value outliers, but also plateau-ish value courses.


    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    thresh : float
        Minimum difference between to values, to consider the latter one as a spike. See condition (1)
    tolerance : float
        Maximum difference between pre-spike and post-spike values. See condition (2)
    window : {str, int}, default '15min'
        Maximum length of "spiky" value courses. See condition (3). Integer defined window length are only allowed for
        regularly sampled timeseries.
    rel_thresh : {float, None}, default None
        Relative threshold.
    numba_kickin : int, default 200000
        When there are detected more than `numba_kickin` incidents of potential spikes,
        the pandas.rolling - part of computation gets "jitted" with numba.
        Default value hast proven to be around the break even point between "jit-boost" and "jit-costs".


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    The implementation is a time-window based version of an outlier test from the UFZ Python library,
    that can be found here:

    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py

    """
    dataseries = data[field].dropna()

    # using reverted series - because ... long story.
    ind = dataseries.index
    rev_ind = ind[0] + ((ind[-1]-ind)[::-1])
    map_i = pd.Series(ind, index=rev_ind)
    dataseries = pd.Series(dataseries.values, index=rev_ind)

    if isinstance(window, int):
        delta = getFreqDelta(dataseries.index)
        window = delta * window
        if not delta:
            raise TypeError('Only offset string defined window sizes allowed for irrgegularily sampled timeseries')

    # get all the entries preceding a significant jump
    if thresh:
        post_jumps = dataseries.diff().abs() > thresh

    if rel_thresh:
        s = np.sign(rel_thresh)
        rel_jumps = s * (dataseries.shift(1).div(dataseries) - 1) > abs(rel_thresh)
        if thresh:
            post_jumps = rel_jumps & post_jumps
        else:
            post_jumps = rel_jumps

    post_jumps = post_jumps[post_jumps]
    if post_jumps.empty:
        return data, flagger

    # get all the entries preceding a significant jump and its successors within "length" range
    to_roll = post_jumps.reindex(dataseries.index, method="bfill", tolerance=window, fill_value=False).dropna()

    if rel_thresh:

        def spikeTester(chunk, thresh=abs(rel_thresh), tol=tolerance):
            jump = chunk[-2] - chunk[-1]
            thresh = thresh * abs(jump)
            chunk_stair = (np.sign(jump) * (chunk - chunk[-1]) < thresh)[::-1].cumsum()
            initial = np.searchsorted(chunk_stair, 2)
            if initial == len(chunk):
                return 0
            if np.abs(chunk[- initial - 1] - chunk[-1]) < tol:
                return initial - 1
            return 0

    else:

        # define spike testing function to roll with (no  rel_check):
        def spikeTester(chunk, thresh=thresh, tol=tolerance):
            # signum change!!!
            chunk_stair = (np.sign(chunk[-2] - chunk[-1]) * (chunk - chunk[-1]) < thresh)[::-1].cumsum()
            initial = np.searchsorted(chunk_stair, 2)
            if initial == len(chunk):
                return 0
            if np.abs(chunk[- initial - 1] - chunk[-1]) < tol:
                return initial - 1
            return 0

    to_roll = dataseries[to_roll]
    roll_mask = pd.Series(False, index=to_roll.index)
    roll_mask[post_jumps.index] = True

    roller = customRoller(to_roll, window=window, mask=roll_mask, min_periods=2, closed='both')
    engine = None if roll_mask.sum() < numba_kickin else 'numba'
    result = roller.apply(spikeTester, raw=True, engine=engine)
    result.index = map_i[result.index]

    # correct the result: only those values define plateaus, that do not have
    # values at their left starting point, that belong to other plateaus themself:
    def calcResult(result):
        var_num = result.shape[0]
        flag_scopes = np.zeros(var_num, dtype=bool)
        for k in range(var_num):
            if result[k] > 0:
                k_r = int(result[k])
                # validity check: plateuas start isnt another plateaus end:
                if not flag_scopes[k - k_r - 1]:
                    flag_scopes[(k - k_r):k] = True
        return pd.Series(flag_scopes, index=result.index)

    cresult = calcResult(result)
    cresult = cresult[cresult].index
    flagger[cresult, field] = kwargs['flag']
    return data, flagger


@register(masking='field', module="outliers")
def flagByGrubbs(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        winsz: Union[FreqString, IntegerWindow],
        alpha: float=0.05,
        min_periods: int=8,
        check_lagged: bool=False,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    The function flags values that are regarded outliers due to the grubbs test.

    See reference [1] for more information on the grubbs tests definition.

    The (two-sided) test gets applied onto data chunks of size "winsz". The tests application  will
    be iterated on each data-chunk under test, till no more outliers are detected in that chunk.

    Note, that the test performs poorely for small data chunks (resulting in heavy overflagging).
    Therefor you should select "winsz" so that every window contains at least > 8 values and also
    adjust the min_periods values accordingly.

    Note, that the data to be tested by the grubbs test are expected to be distributed "normalish".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    winsz : {int, str}
        The size of the window you want to use for outlier testing. If an integer is passed, the size
        refers to the number of periods of every testing window. If a string is passed, it has to be an offset string,
        and will denote the total temporal extension of every window.
    alpha : float, default 0.05
        The level of significance, the grubbs test is to be performed at. (between 0 and 1)
    min_periods : int, default 8
        The minimum number of values that have to be present in an interval under test, for a grubbs test result to be
        accepted. Only makes sence in case `winsz` is an offset string.
    check_lagged: boolean, default False
        If True, every value gets checked twice for being an outlier. Ones in the initial rolling window and one more
        time in a rolling window that is lagged by half the windows delimeter (winsz/2). Recommended for avoiding false
        positives at the window edges. Only available when rolling with integer defined window size.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    References
    ----------
    introduction to the grubbs test:

    [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
    """
    data = data.copy()
    datcol = data[field]
    rate = getFreqDelta(datcol.index)

    # if timeseries that is analyzed, is regular, window size can be transformed to a number of periods:
    if rate and isinstance(winsz, str):
        winsz = pd.Timedelta(winsz) // rate

    to_group = pd.DataFrame(data={"ts": datcol.index, "data": datcol})
    to_flag = pd.Series(False, index=datcol.index)

    # period number defined test intervals
    if isinstance(winsz, int):
        grouper_series = pd.Series(data=np.arange(0, datcol.shape[0]), index=datcol.index)
        grouper_series_lagged = grouper_series + (winsz / 2)
        grouper_series = grouper_series.transform(lambda x: x // winsz)
        grouper_series_lagged = grouper_series_lagged.transform(lambda x: x // winsz)
        partitions = to_group.groupby(grouper_series)
        partitions_lagged = to_group.groupby(grouper_series_lagged)

    # offset defined test intervals:
    else:
        partitions = to_group.groupby(pd.Grouper(freq=winsz))

    for _, partition in partitions:
        if partition.shape[0] > min_periods:
            detected = smirnov_grubbs.two_sided_test_indices(partition["data"].values, alpha=alpha)
            detected = partition["ts"].iloc[detected]
            to_flag[detected.index] = True

    if isinstance(winsz, int) and check_lagged:
        to_flag_lagged = pd.Series(False, index=datcol.index)

        for _, partition in partitions_lagged:
            if partition.shape[0] > min_periods:
                detected = smirnov_grubbs.two_sided_test_indices(partition["data"].values, alpha=alpha)
                detected = partition["ts"].iloc[detected]
                to_flag_lagged[detected.index] = True

        to_flag &= to_flag_lagged

    flagger[to_flag, field] = kwargs['flag']
    return data, flagger


@register(masking='field', module="outliers")
def flagRange(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        min: float=-np.inf,
        max: float=np.inf,
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function flags values not covered by the closed interval [`min`, `max`].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    min : float
        Lower bound for valid data.
    max : float
        Upper bound for valid data.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.
    """

    # using .values is much faster
    datacol = data[field].values
    mask = (datacol < min) | (datacol > max)
    flagger[mask, field] = kwargs['flag']
    return data, flagger


@register(masking='all', module="outliers")
def flagCrossStatistic(
        data: DictOfSeries,
        field: ColumnName,
        flagger: Flagger,
        fields: Sequence[ColumnName],
        thresh: float,
        cross_stat: Literal["modZscore", "Zscore"]="modZscore",
        **kwargs
) -> Tuple[DictOfSeries, Flagger]:
    """
    Function checks for outliers relatively to the "horizontal" input data axis.

    For `fields` :math:`=[f_1,f_2,...,f_N]` and timestamps :math:`[t_1,t_2,...,t_K]`, the following steps are taken
    for outlier detection:

    1. All timestamps :math:`t_i`, where there is one :math:`f_k`, with :math:`data[f_K]` having no entry at
       :math:`t_i`, are excluded from the following process (inner join of the :math:`f_i` fields.)
    2. for every :math:`0 <= i <= K`, the value
       :math:`m_j = median(\\{data[f_1][t_i], data[f_2][t_i], ..., data[f_N][t_i]\\})` is calculated
    2. for every :math:`0 <= i <= K`, the set
       :math:`\\{data[f_1][t_i] - m_j, data[f_2][t_i] - m_j, ..., data[f_N][t_i] - m_j\\}` is tested for outliers with the
       specified method (`cross_stat` parameter).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        A dummy parameter.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional informations related to `data`.
    fields : str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    thresh : float
        Threshold which the outlier score of an value must exceed, for being flagged an outlier.
    cross_stat : {'modZscore', 'Zscore'}, default 'modZscore'
        Method used for calculating the outlier scores.

        * ``'modZscore'``: Median based "sigma"-ish approach. See Referenecs [1].
        * ``'Zscore'``: Score values by how many times the standard deviation they differ from the median.
          See References [1]

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the input flagger.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """

    df = data[fields].loc[data[fields].index_of('shared')].to_df()

    if isinstance(cross_stat, str):

        if cross_stat == 'modZscore':
            MAD_series = df.subtract(df.median(axis=1), axis=0).abs().median(axis=1)
            diff_scores = (0.6745 * (df.subtract(df.median(axis=1), axis=0))).divide(MAD_series, axis=0).abs()

        elif cross_stat == 'Zscore':
            diff_scores = df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0).abs()

        else:
            raise ValueError(cross_stat)

    else:

        try:
            stat = getattr(df, cross_stat.__name__)(axis=1)
        except AttributeError:
            stat = df.aggregate(cross_stat, axis=1)

        diff_scores = df.subtract(stat, axis=0).abs()

    mask = diff_scores > thresh
    for var in fields:
        flagger[mask[var], var] = kwargs['flag']

    return data, flagger
