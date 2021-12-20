#! /usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
from typing import Optional, Union, Tuple, Sequence, Callable
from typing_extensions import Literal

import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd

from dios import DictOfSeries
from outliers import smirnov_grubbs
from scipy.optimize import curve_fit

from saqc.constants import BAD, UNFLAGGED
from saqc.core import register, Flags
from saqc.core.register import flagging
from saqc.lib.tools import customRoller, findIndex, getFreqDelta, toSequence
from saqc.funcs.scores import assignKNNScore
from saqc.funcs.tools import copyField, dropField
from saqc.funcs.transformation import transform
import saqc.lib.ts_operators as ts_ops


@flagging()
def flagByStray(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    freq: Optional[Union[int, str]] = None,
    min_periods: int = 11,
    iter_start: float = 0.5,
    alpha: float = 0.05,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Flag outliers in 1-dimensional (score) data with the STRAY Algorithm.

    Find more information on the algorithm in References [1].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store quality flags to data.

    freq : str, int, or None, default None
        Determines the segmentation of the data into partitions, the kNN algorithm is
        applied onto individually.

        * ``np.inf``: Apply Scoring on whole data set at once
        * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
        * Offset String : Apply scoring on successive partitions of temporal extension
          matching the passed offset string

    min_periods : int, default 11
        Minimum number of periods per partition that have to be present for a valid
        outlier dettection to be made in this partition. (Only of effect, if `freq`
        is an integer.) Partition min value must always be greater then the
        nn_neighbors value.

    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered
        "normal". 0.5 results in the stray algorithm to search only the upper 50 % of
        the scores for the cut off point. (See reference section for more information)

    alpha : float, default 0.05
        Level of significance by which it is tested, if a score might be drawn from
        another distribution, than the majority of the data.

    flag : float, default BAD
        flag to set.

    References
    ----------
    [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in
        high dimensional data. arXiv preprint arXiv:1908.04000.
    """
    scores = data[field].dropna()

    if scores.empty:
        return data, flags

    if not freq:
        freq = scores.shape[0]

    if isinstance(freq, str):
        partitions = scores.groupby(pd.Grouper(freq=freq))

    else:
        grouper_series = pd.Series(
            data=np.arange(0, scores.shape[0]), index=scores.index
        )
        grouper_series = grouper_series.transform(lambda x: int(np.floor(x / freq)))
        partitions = scores.groupby(grouper_series)

    # calculate flags for every partition
    for _, partition in partitions:

        if partition.empty | (partition.shape[0] < min_periods):
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
                flags[index, field] = flag
                break

    return data, flags


def _evalStrayLabels(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    target: Sequence[str],
    reduction_range: Optional[str] = None,
    reduction_drop_flagged: bool = False,  # TODO: still a case ?
    reduction_thresh: float = 3.5,
    reduction_min_periods: int = 1,
    at_least_one: bool = True,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function "reduces" an observations flag to components of it, by applying MAD
    (See references) test onto every components temporal surrounding.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : str
        The fieldname of the column, holding the labels to be evaluated.

    flags : saqc.Flags
        Container to store quality flags to data.

    target : list of str
        A list of strings, holding the column names of the variables, the stray labels
        shall be projected onto.

    val_frame : (N,M) pd.DataFrame
        Input NxM DataFrame of observations, where N is the number of observations and
        M the number of components per observation.

    to_flag_frame : pandas.DataFrame
        Input dataframe of observations to be tested, where N is the number of
        observations and M the number of components per observation.

    reduction_range : {None, str}
        An offset string, denoting the range of the temporal surrounding to include
        into the MAD testing. If ``None`` is passed, no testing will be performed and
        all targets will have the stray flag projected.

    reduction_drop_flagged : bool, default False
        Wheather or not to drop flagged values other than the value under test, from the
        temporal surrounding before checking the value with MAD.

    reduction_thresh : float, default 3.5
        The `critical` value, controlling wheather the MAD score is considered
        referring to an outlier or not. Higher values result in less rigid flagging.
        The default value is widely used in the literature. See references section
        for more details ([1]).

    at_least_one : bool, default True
        If none of the variables, the outlier label shall be reduced to, is an outlier
        with regard to the test, all (True) or none (False) of the variables are flagged

    flag : float, default BAD
        flag to set.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    val_frame = data[target].to_df()
    stray_detects = flags[field] > UNFLAGGED
    stray_detects = stray_detects[stray_detects]
    to_flag_frame = pd.DataFrame(False, columns=target, index=stray_detects.index)

    if reduction_range is None:
        for field in to_flag_frame.columns:
            flags[to_flag_frame.index, field] = flag
        return data, flags

    for var in target:
        for index in enumerate(to_flag_frame.index):

            index_slice = slice(
                index[1] - pd.Timedelta(reduction_range),
                index[1] + pd.Timedelta(reduction_range),
            )
            test_slice = val_frame[var][index_slice].dropna()

            # check, wheather value under test is sufficiently centered:
            first = test_slice.first_valid_index()
            last = test_slice.last_valid_index()
            min_range = pd.Timedelta(reduction_range) / 4

            if (
                pd.Timedelta(index[1] - first) < min_range
                or pd.Timedelta(last - index[1]) < min_range
            ):
                polydeg = 0
            else:
                polydeg = 2

            if reduction_drop_flagged:
                test_slice = test_slice.drop(to_flag_frame.index, errors="ignore")

            if test_slice.shape[0] < reduction_min_periods:
                to_flag_frame.loc[index[1], var] = True
                continue

            x = test_slice.index.values.astype(float)
            x_0 = x[0]
            x = (x - x_0) / 10 ** 12

            polyfitted = poly.polyfit(y=test_slice.values, x=x, deg=polydeg)

            testval = poly.polyval(
                (float(index[1].to_numpy()) - x_0) / 10 ** 12, polyfitted
            )
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
        flags[col[col].index, field] = flag

    return data, flags


def _expFit(
    val_frame,
    scoring_method="kNNMaxGap",
    n_neighbors=10,
    iter_start=0.5,
    alpha=0.05,
    bin_frac=10,
):
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
        `'kNNSum'`: Assign to every point the sum of the distances to its 'n' nearest neighbors.
        `'kNNMaxGap'`: Assign to every point the distance to the neighbor with the "maximum gap" to its predecessor
        in the hierarchy of the `n` nearest neighbors. (see reference section for further descriptions)
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
        binz = np.linspace(
            resids[0], resids[-1], 10 * int(np.ceil(data_len / bin_frac))
        )
    elif bin_frac in [
        "auto",
        "fd",
        "doane",
        "scott",
        "stone",
        "rice",
        "sturges",
        "sqrt",
    ]:
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
            "or a too low value for 'offset' was passed "
            "(offset better be much greater 0.5)"
        )
    # GO!
    iter_max_bin_index = findIndex(binz, resids[iter_index - 1], 0)
    upper_tail_index = int(np.floor(0.5 * hist_argmax + 0.5 * iter_max_bin_index))
    resids_tail_index = findIndex(resids, binz[upper_tail_index], 0)
    upper_tail_hist, bins = np.histogram(
        resids[resids_tail_index:iter_index],
        bins=binz[upper_tail_index : iter_max_bin_index + 1],
    )

    while (test_val < crit_val) & (iter_index < resids.size - 1):
        iter_index += 1
        new_iter_max_bin_index = findIndex(binz, resids[iter_index - 1], 0)
        # following if/else block "manually" expands the data histogram and circumvents calculation of the complete
        # histogram in any new iteration.
        if new_iter_max_bin_index == iter_max_bin_index:
            upper_tail_hist[-1] += 1
        else:
            upper_tail_hist = np.append(
                upper_tail_hist, np.zeros([new_iter_max_bin_index - iter_max_bin_index])
            )
            upper_tail_hist[-1] += 1
            iter_max_bin_index = new_iter_max_bin_index
            upper_tail_index_new = int(
                np.floor(0.5 * hist_argmax + 0.5 * iter_max_bin_index)
            )
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


@register(
    mask=["field"],
    demask=["field"],
    squeeze=["field"],
    multivariate=True,
    handles_target=False,
)
def flagMVScores(
    data: DictOfSeries,
    field: Sequence[str],
    flags: Flags,
    trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
    alpha: float = 0.05,
    n: int = 10,
    func: Callable[[pd.Series], float] = np.sum,
    iter_start: float = 0.5,
    partition: Optional[Union[int, str]] = None,
    partition_min: int = 11,
    stray_range: Optional[str] = None,
    drop_flagged: bool = False,  # TODO: still a case ?
    thresh: float = 3.5,
    min_periods: int = 1,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The algorithm implements a 3-step outlier detection procedure for simultaneously
    flagging of higher dimensional data (dimensions > 3).

    In references [1], the procedure is introduced and exemplified with an
    application on hydrological data. See the notes section for an overview over the
    algorithms basic steps.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.

    field : list of str
        List of fieldnames, corresponding to the variables that are to be included
        into the flagging process.

    flags : saqc.Flags
        Container to store quality flags to data.

    trafo : callable, default lambda x:x
        Transformation to be applied onto every column before scoring. Will likely
        get deprecated soon. Its better to transform the data in a processing step,
        preceeeding the call to ``flagMVScores``.

    alpha : float, default 0.05
        Level of significance by which it is tested, if an observations score might
        be drawn from another distribution than the majority of the observation.

    n : int, default 10
        Number of neighbors included in the scoring process for every datapoint.

    func : Callable[numpy.array, float], default np.sum
        The function that maps the set of every points k-nearest neighbor distances
        onto a certain scoring.

    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered
        "normal". 0.5 results in the threshing algorithm to search only the upper 50
        % of the scores for the cut off point. (See reference section for more
        information)

    partition : {None, str, int}, default None
        Only effective when `threshing` = 'stray'. Determines the size of the data
        partitions, the data is decomposed into. Each partition is checked seperately
        for outliers. If a String is passed, it has to be an offset string and it
        results in partitioning the data into parts of according temporal length. If
        an integer is passed, the data is simply split up into continous chunks of
        `freq` periods. if ``None`` is passed (default), all the data will be tested
        in one run.

    partition_min : int, default 11
        Only effective when `threshing` = 'stray'. Minimum number of periods per
        partition that have to be present for a valid outlier detection to be made in
        this partition. (Only of effect, if `stray_partition` is an integer.)

    partition_trafo : bool, default True
        Whether or not to apply the passed transformation on every partition the
        algorithm is applied on, separately.

    stray_range : {None, str}, default None
        If not None, it is tried to reduce the stray result onto single outlier
        components of the input fields. An offset string, denoting the range of the
        temporal surrounding to include into the MAD testing while trying to reduce
        flags.

    drop_flagged : bool, default False
        Only effective when `range` is not ``None``. Whether or not to drop flagged
        values other than the value under test from the temporal surrounding before
        checking the value with MAD.

    thresh : float, default 3.5
        Only effective when `range` is not ``None``. The `critical` value,
        controlling wheather the MAD score is considered referring to an outlier or
        not. Higher values result in less rigid flagging. The default value is widely
        considered apropriate in the literature.

    min_periods : int, 1
        Only effective when `range` is not ``None``. Minimum number of meassurements
        necessarily present in a reduction interval for reduction actually to be
        performed.

    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed, relatively to the flags input.

    Notes
    -----
    The basic steps are:

    1. transforming

    The different data columns are transformed via timeseries transformations to
    (a) make them comparable and
    (b) make outliers more stand out.

    This step is usually subject to a phase of research/try and error. See [1] for more
    details.

    Note, that the data transformation as an built-in step of the algorithm,
    will likely get deprecated soon. Its better to transform the data in a processing
    step, preceeding the multivariate flagging process. Also, by doing so, one gets
    mutch more control and variety in the transformation applied, since the `trafo`
    parameter only allows for application of the same transformation to all of the
    variables involved.

    2. scoring

    Every observation gets assigned a score depending on its k nearest neighbors. See
    the `scoring_method` parameter description for details on the different scoring
    methods. Furthermore [1], [2] may give some insight in the pro and cons of the
    different methods.

    3. threshing

    The gaps between the (greatest) scores are tested for beeing drawn from the same
    distribution as the majority of the scores. If a gap is encountered, that,
    with sufficient significance, can be said to not be drawn from the same
    distribution as the one all the smaller gaps are drawn from, than the observation
    belonging to this gap, and all the observations belonging to gaps larger then
    this gap, get flagged outliers. See description of the `threshing` parameter for
    more details. Although [2] gives a fully detailed overview over the `stray`
    algorithm.
    """

    fields = toSequence(field)

    fields_ = []
    for f in fields:
        field_ = str(uuid.uuid4())
        data, flags = copyField(data, field=f, flags=flags, target=field_)
        data, flags = transform(
            data, field=field_, flags=flags, func=trafo, freq=partition
        )
        fields_.append(field_)

    knn_field = str(uuid.uuid4())
    data, flags = assignKNNScore(
        data=data,
        field=fields_,
        flags=flags,
        target=knn_field,
        n=n,
        func=func,
        freq=partition,
        method="ball_tree",
        min_periods=partition_min,
        **kwargs,
    )
    for field_ in fields_:
        data, flags = dropField(data, field_, flags)

    data, flags = flagByStray(
        data=data,
        field=knn_field,
        flags=flags,
        freq=partition,
        min_periods=partition_min,
        iter_start=iter_start,
        alpha=alpha,
        flag=flag,
        **kwargs,
    )

    data, flags = _evalStrayLabels(
        data=data,
        field=knn_field,
        target=fields,
        flags=flags,
        reduction_range=stray_range,
        reduction_drop_flagged=drop_flagged,
        reduction_thresh=thresh,
        reduction_min_periods=min_periods,
        flag=flag,
        **kwargs,
    )
    data, flags = dropField(data, knn_field, flags)

    return data, flags


@flagging()
def flagRaise(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    thresh: float,
    raise_window: str,
    freq: str,
    average_window: Optional[str] = None,
    raise_factor: float = 2.0,
    slope: Optional[float] = None,
    weight: float = 0.8,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
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
    flags : saqc.Flags
        Container to store flags of the data.
    thresh : float
        The threshold, for the total rise (thresh > 0), or total drop (thresh < 0),
        value courses must not exceed within a timespan of length `raise_window`.
    raise_window : str
        An offset string, determining the timespan, the rise/drop thresholding refers
        to. Window is inclusively defined.
    freq : str
        An offset string, determining The frequency, the timeseries to-be-flagged is
        supposed to be sampled at. The window is inclusively defined.
    average_window : {None, str}, default None
        See condition (2) of the description linked in the references. Window is
        inclusively defined. The window defaults to 1.5 times the size of `raise_window`
    raise_factor : float, default 2
        See second condition listed in the notes below.
    slope : {None, float}, default None
        See third condition listed in the notes below.
    weight : float, default 0.8
        See third condition listed in the notes below.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed, relatively to the flags input.

    Notes
    -----
    The value :math:`x_{k}` of a time series :math:`x` with associated
    timestamps :math:`t_i`, is flagged a raise, if:

    * There is any value :math:`x_{s}`, preceeding :math:`x_{k}` within `raise_window`
      range, so that:

      * :math:`M = |x_k - x_s | >`  `thresh` :math:`> 0`

    * The weighted average :math:`\\mu^{*}` of the values, preceding :math:`x_{k}`
      within `average_window`
      range indicates, that :math:`x_{k}` does not return from an "outlierish" value
      course, meaning that:

      * :math:`x_k > \\mu^* + ( M` / `mean_raise_factor` :math:`)`

    * Additionally, if ``min_slope`` is not `None`, :math:`x_{k}` is checked for being
      sufficiently divergent from its very predecessor :math:`x_{k-1}`, meaning that, it
      is additionally checked if:

      * :math:`x_k - x_{k-1} >` `min_slope`
      * :math:`t_k - t_{k-1} >` `weight` :math:`\\times` `freq`

    """

    # prepare input args
    dataseries = data[field].dropna()
    raise_window = pd.Timedelta(raise_window)
    freq = pd.Timedelta(freq)
    if slope is not None:
        slope = np.abs(slope)

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

    numba_boost = True
    if numba_boost:
        raise_check = numba.jit(raise_check, nopython=True)
        raise_series = raise_series.apply(
            raise_check, args=(thresh,), raw=True, engine="numba"
        )
    else:
        raise_series = raise_series.apply(raise_check, args=(thresh,), raw=True)

    if raise_series.isna().all():
        return data, flags

    # "unflag" values of insufficient deviation to their predecessors
    if slope is not None:
        w_mask = (
            pd.Series(dataseries.index).diff().dt.total_seconds() / freq.total_seconds()
        ) > weight
        slope_mask = np.abs(dataseries.diff()) < slope
        to_unflag = raise_series.notna() & w_mask.values & slope_mask
        raise_series[to_unflag] = np.nan

    # calculate and apply the weighted mean weights (pseudo-harmonization):
    weights = (
        pd.Series(dataseries.index).diff(periods=2).shift(-1).dt.total_seconds()
        / freq.total_seconds()
        / 2
    )

    weights.iloc[0] = 0.5 + (
        dataseries.index[1] - dataseries.index[0]
    ).total_seconds() / (freq.total_seconds() * 2)

    weights.iloc[-1] = 0.5 + (
        dataseries.index[-1] - dataseries.index[-2]
    ).total_seconds() / (freq.total_seconds() * 2)

    weights[weights > 1.5] = 1.5
    weights.index = dataseries.index
    weighted_data = dataseries.mul(weights)

    # rolling weighted mean calculation
    weighted_rolling_mean = weighted_data.rolling(
        average_window, min_periods=2, closed="both"
    )
    weights_rolling_sum = weights.rolling(average_window, min_periods=2, closed="both")
    if numba_boost:
        custom_rolling_mean = numba.jit(custom_rolling_mean, nopython=True)
        weighted_rolling_mean = weighted_rolling_mean.apply(
            custom_rolling_mean, raw=True, engine="numba"
        )
        weights_rolling_sum = weights_rolling_sum.apply(
            custom_rolling_mean, raw=True, engine="numba"
        )
    else:
        weighted_rolling_mean = weighted_rolling_mean.apply(
            custom_rolling_mean, raw=True
        )
        weights_rolling_sum = weights_rolling_sum.apply(
            custom_rolling_mean, raw=True, engine="numba"
        )

    weighted_rolling_mean = weighted_rolling_mean / weights_rolling_sum
    # check means against critical raise value:
    to_flag = dataseries >= weighted_rolling_mean + (raise_series / raise_factor)
    to_flag &= raise_series.notna()
    flags[to_flag[to_flag].index, field] = flag

    return data, flags


@flagging()
def flagMAD(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: str,
    z: float = 3.5,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
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
    flags : saqc.Flags
        Container to store flags of the data.
    window : str
       Offset string. Denoting the windows size that the "Z-scored" values have to lie in.
    z: float, default 3.5
        The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed, relatively to the flags input.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    d = data[field]
    if d.empty:
        return data, flags

    median = d.rolling(window=window, closed="both").median()
    diff = (d - median).abs()
    mad = diff.rolling(window=window, closed="both").median()
    mask = (mad > 0) & (0.6745 * diff > z * mad)
    # NOTE:
    # In pandas <= 0.25.3, the window size is not fixed if the
    # window-argument to rolling is a frequency. That implies,
    # that during the first iterations the window has a size of
    # 1, 2, 3, ... until it eventually covers the desired time
    # span. For stuff like the calculation of median, that is rather
    # unfortunate, as the size of the calculation base might differ
    # heavily. So don't flag something until, the window reaches
    # its target size
    if not isinstance(window, int):
        index = mask.index
        mask.loc[index < index[0] + pd.to_timedelta(window)] = False

    flags[mask, field] = flag
    return data, flags


@flagging()
def flagOffset(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    thresh: float,
    tolerance: float,
    window: Union[int, str],
    thresh_relative: Optional[float] = None,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    A basic outlier test that work on regular and irregular sampled data

    The test classifies values/value courses as outliers by detecting not only a rise
    in value, but also, checking for a return to the initial value level.

    Values :math:`x_n, x_{n+1}, .... , x_{n+k}` of a timeseries :math:`x` with
    associated timestamps :math:`t_n, t_{n+1}, .... , t_{n+k}` are considered spikes, if

    1. :math:`|x_{n-1} - x_{n + s}| >` `thresh`, for all :math:`s \\in [0,1,2,...,k]`

    2. :math:`|x_{n-1} - x_{n+k+1}| <` `tolerance`

    3. :math:`|t_{n-1} - t_{n+k+1}| <` `window`

    Note, that this definition of a "spike" not only includes one-value outliers, but
    also plateau-ish value courses.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The field in data.
    flags : saqc.Flags
        Container to store flags of the data.
    thresh : float
        Minimum difference between to values, to consider the latter one as a spike. See condition (1)
    tolerance : float
        Maximum difference between pre-spike and post-spike values. See condition (2)
    window : {str, int}, default '15min'
        Maximum length of "spiky" value courses. See condition (3). Integer defined window length are only allowed for
        regularly sampled timeseries.
    thresh_relative : {float, None}, default None
        Relative threshold.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed, relatively to the flags input.

    References
    ----------
    The implementation is a time-window based version of an outlier test from the UFZ Python library,
    that can be found here:

    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py

    """
    dataseries = data[field].dropna()
    if dataseries.empty:
        return data, flags

    # using reverted series - because ... long story.
    ind = dataseries.index
    rev_ind = ind[0] + ((ind[-1] - ind)[::-1])
    map_i = pd.Series(ind, index=rev_ind)
    dataseries = pd.Series(dataseries.values, index=rev_ind)

    if isinstance(window, int):
        delta = getFreqDelta(dataseries.index)
        window = delta * window
        if not delta:
            raise TypeError(
                "Only offset string defined window sizes allowed for irrgegularily sampled timeseries"
            )

    # get all the entries preceding a significant jump
    if thresh:
        post_jumps = dataseries.diff().abs() > thresh

    if thresh_relative:
        s = np.sign(thresh_relative)
        rel_jumps = s * (dataseries.shift(1) - dataseries).div(dataseries.abs()) > abs(
            thresh_relative
        )
        if thresh:
            post_jumps = rel_jumps & post_jumps
        else:
            post_jumps = rel_jumps

    post_jumps = post_jumps[post_jumps]
    if post_jumps.empty:
        return data, flags

    # get all the entries preceding a significant jump
    # and its successors within "length" range
    to_roll = post_jumps.reindex(
        dataseries.index, method="bfill", tolerance=window, fill_value=False
    ).dropna()
    to_roll = dataseries[to_roll]

    if thresh_relative:

        def spikeTester(chunk, thresh=abs(thresh_relative), tol=tolerance):
            jump = chunk[-2] - chunk[-1]
            thresh = thresh * abs(jump)
            chunk_stair = (np.sign(jump) * (chunk - chunk[-1]) < thresh)[::-1].cumsum()
            initial = np.searchsorted(chunk_stair, 2)
            if initial == len(chunk):
                return 0
            if np.abs(chunk[-initial - 1] - chunk[-1]) < tol:
                return initial - 1
            return 0

    else:

        # define spike testing function to roll with (no  rel_check):
        def spikeTester(chunk, thresh=thresh, tol=tolerance):
            # signum change!!!
            chunk_stair = (
                np.sign(chunk[-2] - chunk[-1]) * (chunk - chunk[-1]) < thresh
            )[::-1].cumsum()
            initial = np.searchsorted(chunk_stair, 2)
            if initial == len(chunk):
                return 0
            if np.abs(chunk[-initial - 1] - chunk[-1]) < tol:
                return initial - 1
            return 0

    roller = customRoller(to_roll, window=window, min_periods=2, closed="both")
    engine = None if len(to_roll) < 200000 else "numba"
    result = roller.apply(spikeTester, raw=True, engine=engine)

    ignore = pd.Series(True, index=to_roll.index)
    ignore[post_jumps.index] = False
    result[ignore] = np.nan

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
                    flag_scopes[(k - k_r) : k] = True
        return pd.Series(flag_scopes, index=result.index)

    cresult = calcResult(result)
    cresult = cresult[cresult].index
    flags[cresult, field] = flag
    return data, flags


@flagging()
def flagByGrubbs(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    window: Union[str, int],
    alpha: float = 0.05,
    min_periods: int = 8,
    pedantic: bool = False,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    The function flags values that are regarded outliers due to the grubbs test.

    See reference [1] for more information on the grubbs tests definition.

    The (two-sided) test gets applied onto data chunks of size "window". The tests
    application  will be iterated on each data-chunk under test, till no more
    outliers are detected in that chunk.

    Note, that the test performs poorely for small data chunks (resulting in heavy
    overflagging). Therefor you should select "window" so that every window contains
    at least > 8 values and also adjust the min_periods values accordingly.

    Note, that the data to be tested by the grubbs test are expected to be distributed
    "normalish".

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store flags of the data.
    window : {int, str}
        The size of the window you want to use for outlier testing. If an integer is
        passed, the size refers to the number of periods of every testing window. If a
        string is passed, it has to be an offset string, and will denote the total
        temporal extension of every window.
    alpha : float, default 0.05
        The level of significance, the grubbs test is to be performed at. (between 0 and 1)
    min_periods : int, default 8
        The minimum number of values that have to be present in an interval under test,
        for a grubbs test result to be accepted. Only makes sence in case `window` is
        an offset string.
    pedantic: boolean, default False
        If True, every value gets checked twice for being an outlier. Ones in the
        initial rolling window and one more time in a rolling window that is lagged
        by half the windows delimeter (window/2). Recommended for avoiding false
        positives at the window edges. Only available when rolling with integer
        defined window size.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the flags input.

    References
    ----------
    introduction to the grubbs test:

    [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
    """
    datcol = data[field].copy()
    rate = getFreqDelta(datcol.index)

    # if timeseries that is analyzed, is regular,
    # window size can be transformed to a number of periods:
    if rate and isinstance(window, str):
        window = pd.Timedelta(window) // rate

    to_group = pd.DataFrame(data={"ts": datcol.index, "data": datcol})
    to_flag = pd.Series(False, index=datcol.index)

    # period number defined test intervals
    if isinstance(window, int):
        grouper_series = pd.Series(
            data=np.arange(0, datcol.shape[0]), index=datcol.index
        )
        grouper_series_lagged = grouper_series + (window / 2)
        grouper_series = grouper_series.transform(lambda x: x // window)
        grouper_series_lagged = grouper_series_lagged.transform(lambda x: x // window)
        partitions = to_group.groupby(grouper_series)
        partitions_lagged = to_group.groupby(grouper_series_lagged)

    # offset defined test intervals:
    else:
        partitions = to_group.groupby(pd.Grouper(freq=window))
        partitions_lagged = []

    for _, partition in partitions:
        if partition.shape[0] > min_periods:
            detected = smirnov_grubbs.two_sided_test_indices(
                partition["data"].values, alpha=alpha
            )
            detected = partition["ts"].iloc[detected]
            to_flag[detected.index] = True

    if isinstance(window, int) and pedantic:
        to_flag_lagged = pd.Series(False, index=datcol.index)

        for _, partition in partitions_lagged:
            if partition.shape[0] > min_periods:
                detected = smirnov_grubbs.two_sided_test_indices(
                    partition["data"].values, alpha=alpha
                )
                detected = partition["ts"].iloc[detected]
                to_flag_lagged[detected.index] = True

        to_flag &= to_flag_lagged

    flags[to_flag, field] = flag
    return data, flags


@flagging()
def flagRange(
    data: DictOfSeries,
    field: str,
    flags: Flags,
    min: float = -np.inf,
    max: float = np.inf,
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Function flags values not covered by the closed interval [`min`, `max`].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flags : saqc.Flags
        Container to store flags of the data.
    min : float
        Lower bound for valid data.
    max : float
        Upper bound for valid data.
    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
    """

    # using .values is much faster
    datacol = data[field].values
    mask = (datacol < min) | (datacol > max)
    flags[mask, field] = flag
    return data, flags


@register(
    mask=["field"],
    demask=["field"],
    squeeze=["field"],
    multivariate=True,
    handles_target=False,
)
def flagCrossStatistics(
    data: DictOfSeries,
    field: Sequence[str],
    flags: Flags,
    thresh: float,
    method: Literal["modZscore", "Zscore"] = "modZscore",
    flag: float = BAD,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Function checks for outliers relatively to the "horizontal" input data axis.

    For `fields` :math:`=[f_1,f_2,...,f_N]` and timestamps :math:`[t_1,t_2,...,t_K]`, the following steps are taken
    for outlier detection:

    1. All timestamps :math:`t_i`, where there is one :math:`f_k`, with :math:`data[f_K]` having no entry at
       :math:`t_i`, are excluded from the following process (inner join of the :math:`f_i` fields.)
    2. for every :math:`0 <= i <= K`, the value
       :math:`m_j = median(\\{data[f_1][t_i], data[f_2][t_i], ..., data[f_N][t_i]\\})` is calculated
    3. for every :math:`0 <= i <= K`, the set
       :math:`\\{data[f_1][t_i] - m_j, data[f_2][t_i] - m_j, ..., data[f_N][t_i] - m_j\\}` is tested for outliers with the
       specified method (`cross_stat` parameter).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : list of str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    flags : saqc.Flags
        A flags object, holding flags and additional informations related to `data`.
    thresh : float
        Threshold which the outlier score of an value must exceed, for being flagged an outlier.
    method : {'modZscore', 'Zscore'}, default 'modZscore'
        Method used for calculating the outlier scores.

        * ``'modZscore'``: Median based "sigma"-ish approach. See Referenecs [1].
        * ``'Zscore'``: Score values by how many times the standard deviation they differ from the median.
          See References [1]

    flag : float, default BAD
        flag to set.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the input flags.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """

    fields = toSequence(field)

    for src in fields[1:]:
        if (data[src].index != data[fields[0]].index).any():
            raise ValueError(
                f"indices of '{fields[0]}' and '{src}' are not compatibble, "
                "please resample all variables to a common (time-)grid"
            )

    df = data[fields].loc[data[fields].index_of("shared")].to_df()

    if isinstance(method, str):

        if method == "modZscore":
            MAD_series = df.subtract(df.median(axis=1), axis=0).abs().median(axis=1)
            diff_scores = (
                (0.6745 * (df.subtract(df.median(axis=1), axis=0)))
                .divide(MAD_series, axis=0)
                .abs()
            )

        elif method == "Zscore":
            diff_scores = (
                df.subtract(df.mean(axis=1), axis=0)
                .divide(df.std(axis=1), axis=0)
                .abs()
            )

        else:
            raise ValueError(method)

    else:

        try:
            stat = getattr(df, method.__name__)(axis=1)
        except AttributeError:
            stat = df.aggregate(method, axis=1)

        diff_scores = df.subtract(stat, axis=0).abs()

    mask = diff_scores > thresh
    if mask.empty:
        return data, flags

    for f in fields:
        flags[mask[f], f] = flag

    return data, flags
