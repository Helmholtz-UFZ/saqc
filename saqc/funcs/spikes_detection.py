#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from scipy.optimize import curve_fit
from saqc.core.register import register
import numpy.polynomial.polynomial as poly
import numba
import saqc.lib.ts_operators as ts_ops
from saqc.lib.tools import (
    retrieveTrustworthyOriginal,
    offset2seconds,
    slidingWindowIndices,
    findIndex,
    toSequence,
    customRoller
)
from outliers import smirnov_grubbs

def _stray(
    val_frame,
    partition_freq=None,
    partition_min=11,
    scoring_method="kNNMaxGap",
    n_neighbors=10,
    iter_start=0.5,
    alpha=0.05,
    trafo=lambda x: x

):
    """
    Find outliers in multi dimensional observations.

    The general idea is to assigning scores to every observation based on the observations neighborhood in the space
    of observations. Then, the gaps between the (greatest) scores are tested for beeing drawn from the same
    distribution, as the majority of the scores.

    See the References section for a link to a detailed description of the algorithm.

    Note, that the flagging result depends on the size of the partition under test and the distribution of the outliers
    in it. For "normalish" and/or slightly "erratic" datasets, 5000 - 10000, periods turned out to be a good guess.

    Note, that no normalizations/transformations are applied to the different components (data columns)
    - those are expected to be applied previously, if necessary.

    Parameters
    ----------
    val_frame : (N,M) ndarray
        Input NxM array of observations, where N is the number of observations and M the number of components per
        observation.
    partition_freq : {None, str, int}, default None
        Determines the size of the data partitions, the data is decomposed into. Each partition is checked seperately
        for outliers. If a String is passed, it has to be an offset string and it results in partitioning the data into
        parts of according temporal length. If an integer is passed, the data is simply split up into continous chunks
        of `partition_freq` periods. if ``None`` is passed (default), all the data will be tested in one run.
    partition_min : int, default 0
        Minimum number of periods per partition that have to be present for a valid outlier dettection to be made in
        this partition. (Only of effect, if `partition_freq` is an integer.) Partition min value must always be
        greater then the nn_neighbors value.
    scoring_method : {'kNNSum', 'kNNMaxGap'}, default 'kNNMaxGap'
        Scoring method applied.
        `'kNNSum'`: Assign to every point the sum of the distances to its 'n_neighbors' nearest neighbors.
        `'kNNMaxGap'`: Assign to every point the distance to the neighbor with the "maximum gap" to its predecessor
        in the hierarchy of the `n_neighbors` nearest neighbors. (see reference section for further descriptions)
    n_neighbors : int, default 10
        Number of neighbors included in the scoring process for every datapoint.
    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered "normal". 0.5 results in the stray
        algorithm to search only the upper 50 % of the scores for the cut off point. (See reference section for more
        information)
    alpha : float, default 0.05
        Niveau of significance by which it is tested, if a score might be drawn from another distribution, than the
        majority of the data.

    References
    ----------
    Detailed description of the Stray algorithm is covered here:

    [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in high dimensional data.
        arXiv preprint arXiv:1908.04000.
    """

    kNNfunc = getattr(ts_ops, scoring_method)
    # partitioning
    if not partition_freq:
        partition_freq = val_frame.shape[0]

    if isinstance(partition_freq, str):
        partitions = val_frame.groupby(pd.Grouper(freq=partition_freq))
    else:
        grouper_series = pd.Series(data=np.arange(0, val_frame.shape[0]), index=val_frame.index)
        grouper_series = grouper_series.transform(lambda x: int(np.floor(x / partition_freq)))
        partitions = val_frame.groupby(grouper_series)

    # calculate flags for every partition
    to_flag = []
    for _, partition in partitions:
        if partition.empty | (partition.shape[0] < partition_min):
            continue
        partition = partition.apply(trafo)
        sample_size = partition.shape[0]
        nn_neighbors = min(n_neighbors, max(sample_size, 2))
        resids = kNNfunc(partition.values, n_neighbors=nn_neighbors - 1, algorithm="ball_tree")
        sorted_i = resids.argsort()
        resids = resids[sorted_i]
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
                break

        to_flag = np.append(to_flag, list(partition.index[sorted_i[iter_index:]]))

    return to_flag


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


def _reduceMVflags(
    val_frame, fields, flagger, to_flag_frame, reduction_range, reduction_drop_flagged=False, reduction_thresh=3.5,
        reduction_min_periods=1
):
    """
    Function called by "spikes_flagMultivarScores" to reduce the number of false positives that result from
    the algorithms confinement to only flag complete observations (all of its variables/components).

    The function "reduces" an observations flag to components of it, by applying MAD (See references)
    test onto every components temporal surrounding.

    Parameters
    ----------
    val_frame : (N,M) pd.DataFrame
        Input NxM DataFrame of observations, where N is the number of observations and M the number of components per
        observation.
    fields : str
        Fieldnames of the components in `val_frame` that are to be tested for outlierishnes.
    to_flag_frame : (K,M) pd.DataFrame
        Input dataframe of observations to be tested, where N is the number of observations and M the number
        of components per observation.
    reduction_range : str
        An offset string, denoting the range of the temporal surrounding to include into the MAD testing.
    reduction_drop_flagged : bool, default False
        Wheather or not to drop flagged values other than the value under test, from the temporal surrounding
        before checking the value with MAD.
    reduction_thresh : float, default 3.5
        The `critical` value, controlling wheather the MAD score is considered referring to an outlier or not.
        Higher values result in less rigid flagging. The default value is widely used in the literature. See references
        section for more details ([1]).

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """

    to_flag_frame[:] = False
    to_flag_index = to_flag_frame.index
    for var in fields:
        for index in enumerate(to_flag_index):
            index_slice = slice(index[1] - pd.Timedelta(reduction_range), index[1] + pd.Timedelta(reduction_range))

            test_slice = val_frame[var][index_slice].dropna()
            # check, wheather value under test is sufficiently centered:
            first_valid = test_slice.first_valid_index()
            last_valid = test_slice.last_valid_index()
            min_range = pd.Timedelta(reduction_range)/4
            polydeg = 2
            if ((pd.Timedelta(index[1] - first_valid) < min_range) |
                (pd.Timedelta(last_valid - index[1]) < min_range)):
                polydeg = 0
            if reduction_drop_flagged:
                test_slice = test_slice.drop(to_flag_index, errors='ignore')
            if test_slice.shape[0] >= reduction_min_periods:
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
            else:
                to_flag_frame.loc[index[1], var] = True

    return to_flag_frame


@register(masking='all')
def spikes_flagMultivarScores(
    data,
    field,
    flagger,
    fields,
    trafo=np.log,
    alpha=0.05,
    n_neighbors=10,
    scoring_method="kNNMaxGap",
    iter_start=0.5,
    threshing="stray",
    expfit_binning="auto",
    stray_partition=None,
    stray_partition_min=0,
    post_reduction=False,
    reduction_range=None,
    reduction_drop_flagged=False,
    reduction_thresh=3.5,
    reduction_min_periods=1,
    **kwargs,
):
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
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    fields : List[str]
        List of fieldnames, corresponding to the variables that are to be included into the flagging process.
    trafo : callable, default np.log
        Transformation to be applied onto every column before scoring. Will likely get deprecated soon. Its better
        to transform the data in a processing step, preceeeding the call to ``flagMultivarScores``.
    alpha : float, default 0.05
        Level of significance by which it is tested, if an observations score might be drawn from another distribution
        than the majority of the observation.
    n_neighbors : int, default 10
        Number of neighbors included in the scoring process for every datapoint.
    scoring_method : {'kNNSum', 'kNNMaxGap'}, default 'kNNMaxGap'
        Scoring method applied.
        ``'kNNSum'``: Assign to every point the sum of the distances to its 'n_neighbors' nearest neighbors.
        ``'kNNMaxGap'``: Assign to every point the distance to the neighbor with the "maximum gap" to its predecessor
        in the hierarchy of the `n_neighbors` nearest neighbors. (see reference section for further descriptions)
    iter_start : float, default 0.5
        Float in [0,1] that determines which percentage of data is considered "normal". 0.5 results in the threshing
        algorithm to search only the upper 50 % of the scores for the cut off point. (See reference section for more
        information)
    threshing : {'stray', 'expfit'}, default 'stray'
        A string, denoting the threshing algorithm to be applied on the observations scores.
        See the documentations of the algorithms (``_stray``, ``_expfit``) and/or the references sections paragraph [2]
        for more informations on the algorithms.
    expfit_binning : {int, str}, default 'auto'
        Controls the binning for the histogram in the ``expfit`` algorithms fitting step.
        If an integer is passed, the residues will equidistantly be covered by `bin_frac` bins, ranging from the
        minimum to the maximum of the residues. If a string is passed, it will be passed on to the
        ``numpy.histogram_bin_edges`` method.
    stray_partition : {None, str, int}, default None
        Only effective when `threshing` = 'stray'.
        Determines the size of the data partitions, the data is decomposed into. Each partition is checked seperately
        for outliers. If a String is passed, it has to be an offset string and it results in partitioning the data into
        parts of according temporal length. If an integer is passed, the data is simply split up into continous chunks
        of `partition_freq` periods. if ``None`` is passed (default), all the data will be tested in one run.
    stray_partition_min : int, default 0
        Only effective when `threshing` = 'stray'.
        Minimum number of periods per partition that have to be present for a valid outlier detection to be made in
        this partition. (Only of effect, if `stray_partition` is an integer.)
    post_reduction : bool, default False
        Wheather or not it should be tried to reduce the flag of an observation to one or more of its components. See
        documentation of `_reduceMVflags` for more details.
    reduction_range : {None, str}, default None
        Only effective when `post_reduction` = True
        An offset string, denoting the range of the temporal surrounding to include into the MAD testing while trying
        to reduce flags.
    reduction_drop_flagged : bool, default False
        Only effective when `post_reduction` = True
        Wheather or not to drop flagged values other than the value under test from the temporal surrounding
        before checking the value with MAD.
    reduction_thresh : float, default 3.5
        Only effective when `post_reduction` = True
        The `critical` value, controlling wheather the MAD score is considered referring to an outlier or not.
        Higher values result in less rigid flagging. The default value is widely considered apropriate in the
        literature.


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
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

    References
    ----------
    Odd Water Algorithm:

    [1] Talagala, P.D. et al (2019): A Feature-Based Procedure for Detecting Technical Outliers in Water-Quality Data
        From In Situ Sensors. Water Ressources Research, 55(11), 8547-8568.

    A detailed description of the stray algorithm:

    [2] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in high dimensional data.
        arXiv preprint arXiv:1908.04000.

    A detailed description of the MAD outlier scoring:

    [3] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """

    # data fransformation/extraction
    data = data.copy()
    fields = toSequence(fields)
    val_frame = data[fields]
    val_frame = val_frame.loc[val_frame.index_of("shared")].to_df()
    val_frame.dropna(inplace=True)
    val_frame = val_frame.apply(trafo)

    if val_frame.empty:
        return data, flagger

    if threshing == "stray":
        to_flag_index = _stray(
            val_frame,
            partition_freq=stray_partition,
            partition_min=stray_partition_min,
            scoring_method=scoring_method,
            n_neighbors=n_neighbors,
            iter_start=iter_start,
            alpha=alpha
        )

    else:
        val_frame = val_frame.apply(trafo)
        to_flag_index = _expFit(val_frame,
                                scoring_method=scoring_method,
                                n_neighbors=n_neighbors,
                                iter_start=iter_start,
                                alpha=alpha,
                                bin_frac=expfit_binning)

    to_flag_frame = pd.DataFrame({var_name: True for var_name in fields}, index=to_flag_index)
    if post_reduction:
        val_frame = data[toSequence(fields)].to_df()
        to_flag_frame = _reduceMVflags(val_frame, fields, flagger, to_flag_frame, reduction_range,
                                       reduction_drop_flagged=reduction_drop_flagged,
                                       reduction_thresh=reduction_thresh,
                                       reduction_min_periods=reduction_min_periods)


    for var in fields:
        to_flag_ind = to_flag_frame.loc[:, var]
        to_flag_ind = to_flag_ind[to_flag_ind].index
        flagger = flagger.setFlags(var, to_flag_ind, **kwargs)

    return data, flagger


@register(masking='field')
def spikes_flagRaise(
    data,
    field,
    flagger,
    thresh,
    raise_window,
    intended_freq,
    average_window=None,
    mean_raise_factor=2,
    min_slope=None,
    min_slope_weight=0.8,
    numba_boost=True,
    **kwargs,
):
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
    flagger : saqc.flagger.BaseFlagger
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
    flagger : saqc.flagger.BaseFlagger
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
    flagger = flagger.setFlags(field, to_flag[to_flag].index, **kwargs)

    return data, flagger


@register(masking='field')
def spikes_flagSlidingZscore(
    data, field, flagger, window, offset, count=1, polydeg=1, z=3.5, method="modZ", **kwargs,
):
    """
    An outlier detection in a sliding window. The method for detection can be a simple Z-score or the more robust
    modified Z-score, as introduced here [1].

    The steps are:
    1.  a window of size `window` is cut from the data
    2.  the data is fit by a polynomial of the given degree `polydeg`
    3.  the outlier `method` detect potential outlier
    4.  the window is continued by `offset` to the next data-slot.
    5.  processing continue at 1. until end of data.
    6.  all potential outlier, that are detected `count`-many times, are promoted to real outlier and flagged by the `flagger`

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    window: {int, str}
        Integer or offset string (see [2]). The size of the window the outlier detection is run in.
    offset: {int, str}
        Integer or offset string (see [2]). Stepsize the window is set further. default: 1h
    count: int, default 1
        Number of times a value has to be classified an outlier in different windows, to be finally flagged an outlier.
    polydeg : int, default 1
        The degree for the polynomial that is fitted to the data in order to calculate the residues.
    z : float, default 3.5
        The value the (mod.) Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
    method: {'modZ', zscore}, default  'modZ'
        See section `Z-Scores and Modified Z-Scores` in [1].

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    [2] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    """

    use_offset = False
    dx_s = offset
    winsz_s = window
    # check param consistency
    if isinstance(window, str) or isinstance(offset, str):
        if isinstance(window, str) and isinstance(offset, str):
            use_offset = True
            dx_s = offset2seconds(offset)
            winsz_s = offset2seconds(window)
        else:
            raise TypeError(
                f"`window` and `offset` must both be an offset or both be numeric, {window} and {offset} was passed"
            )

    # check params
    if polydeg < 0:
        raise ValueError("polydeg must be positive")
    if z < 0:
        raise ValueError("z must be positive")
    if count <= 0:
        raise ValueError("count must be positive and not zero")

    if dx_s >= winsz_s and count == 1:
        pass
    elif dx_s >= winsz_s and count > 1:
        ValueError("If stepsize `offset` is bigger that the window-size, every value is seen just once, so use count=1")
    elif count > winsz_s // dx_s:
        raise ValueError(
            f"Adjust `offset`, `stepsize` or `window`. A single data point is "
            f"seen `floor(window / offset) = {winsz_s // dx_s}` times, but count is set to {count}"
        )

    # prepare the method
    method = method.lower()
    if method == "modz":

        def _calc(residual):
            diff = np.abs(residual - np.median(residual))
            mad = np.median(diff)
            return (mad > 0) & (0.6745 * diff > z * mad)

    elif method == "zscore":

        def _calc(residual):
            score = zscore(residual, ddof=1)
            return np.abs(score) > z

    else:
        raise NotImplementedError
    method = _calc

    # prepare data, work on numpy arrays for the fulfilling pleasure of performance
    d = data[field].dropna()
    if d.empty:
        return data, flagger
    all_indices = np.arange(len(d.index))
    x = (d.index - d.index[0]).total_seconds().values
    y = d.values
    counters = np.full(len(d.index), count)

    if use_offset:
        _loopfun = slidingWindowIndices
    else:

        def _loopfun(arr, wsz, step):
            for i in range(0, len(arr) - wsz + 1, step):
                yield i, i + wsz

    for start, end in _loopfun(d.index, window, offset):
        # mask points that have been already discarded
        mask = counters[start:end] > 0
        indices = all_indices[all_indices[start:end][mask]]
        xchunk = x[indices]
        ychunk = y[indices]

        if xchunk.size == 0:
            continue

        # get residual
        coef = poly.polyfit(xchunk, ychunk, polydeg)
        model = poly.polyval(xchunk, coef)
        residual = ychunk - model

        score = method(residual)

        # count`em in
        goneMad = score.nonzero()[0]
        counters[indices[goneMad]] -= 1

    outlier = np.where(counters <= 0)[0]
    loc = d[outlier].index
    flagger = flagger.setFlags(field, loc=loc, **kwargs)
    return data, flagger


@register(masking='field')
def spikes_flagMad(data, field, flagger, window, z=3.5, **kwargs):

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
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    window : str
       Offset string. Denoting the windows size that the "Z-scored" values have to lie in.
    z: float, default 3.5
        The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    """
    d = data[field].copy().mask(flagger.isFlagged(field))
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

    flagger = flagger.setFlags(field, mask, **kwargs)
    return data, flagger


@register(masking='field')
def spikes_flagBasic(data, field, flagger, thresh, tolerance, window, numba_kickin=200000, **kwargs):
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
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    thresh : float, default 7
        Minimum difference between to values, to consider the latter one as a spike. See condition (1)
    tolerance : float, default 0
        Maximum difference between pre-spike and post-spike values. See condition (2)
    window : str, default '15min'
        Maximum length of "spiky" value courses. See condition (3)
    numba_kickin : int, default 200000
        When there are detected more than `numba_kickin` incidents of potential spikes,
        the pandas.rolling - part of computation gets "jitted" with numba.
        Default value hast proven to be around the break even point between "jit-boost" and "jit-costs".


    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    The implementation is a time-window based version of an outlier test from the UFZ Python library,
    that can be found here:

    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py

    """

    dataseries = data[field].dropna()
    # get all the entries preceding a significant jump
    post_jumps = dataseries.diff().abs() > thresh
    post_jumps = post_jumps[post_jumps]
    if post_jumps.empty:
        return data, flagger
    # get all the entries preceeding a significant jump and its successors within "length" range
    to_roll = post_jumps.reindex(dataseries.index, method="bfill", tolerance=window, fill_value=False).dropna()

    # define spike testing function to roll with:
    def spikeTester(chunk, thresh=thresh, tol=tolerance):
        # signum change!!!
        chunk_stair = (np.sign(chunk[-2] - chunk[-1])*(chunk - chunk[-1]) < thresh)[::-1].cumsum()
        initial = np.searchsorted(chunk_stair, 2)
        if initial == len(chunk):
            return 0
        if np.abs(chunk[- initial - 1] - chunk[-1]) < tol:
            return initial - 1
        else:
            return 0

    to_roll = dataseries[to_roll]
    roll_mask = pd.Series(False, index=to_roll.index)
    roll_mask[post_jumps.index] = True

    roller = customRoller(to_roll, window=window, mask=roll_mask, min_periods=2, closed='both')
    engine = None if roll_mask.sum() < numba_kickin else 'numba'
    result = roller.apply(spikeTester, raw=True, engine=engine)

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
    flagger = flagger.setFlags(field, cresult, **kwargs)
    return data, flagger


@register(masking='field')
def spikes_flagSpektrumBased(
    data,
    field,
    flagger,
    raise_factor=0.15,
    deriv_factor=0.2,
    noise_func="CoVar",
    noise_window="12h",
    noise_thresh=1,
    smooth_window=None,
    smooth_poly_deg=2,
    **kwargs,
):
    """

    Function detects and flags spikes in input data series by evaluating its derivatives and applying some
    conditions to it. A datapoint is considered a spike, if:

    (1) the quotient to its preceeding datapoint exceeds a certain bound
    (controlled by param `raise_factor`)
    (2) the quotient of the datas second derivate at the preceeding and subsequent timestamps is close enough to 1.
    (controlled by param `deriv_factor`)
    (3) the surrounding data is not too noisy. (Coefficient of Variation[+/- noise_window] < 1)
    (controlled by param `noise_thresh`)

    Note, that the data-to-be-flagged is supposed to be sampled at an equidistant frequency grid

    Note, that the derivative is calculated after applying a Savitsky-Golay filter to the data.

    Parameters
    ----------

    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    raise_factor : float, default 0.15
        Minimum relative value difference between two values to consider the latter as a spike candidate.
        See condition (1) (or reference [2]).
    deriv_factor : float, default 0.2
        See condition (2) (or reference [2]).
    noise_func : {'CoVar', 'rVar'}, default 'CoVar'
        Function to calculate noisiness of the data surrounding potential spikes.

        * ``'CoVar'``: Coefficient of Variation
        * ``'rVar'``: Relative Variance

    noise_window : str, default '12h'
        An offset string that determines the range of the time window of the "surrounding" data of a potential spike.
        See condition (3) (or reference [2]).
    noise_thresh : float, default 1
        Upper threshold for noisiness of data surrounding potential spikes. See condition (3) (or reference [2]).
    smooth_window : {None, str}, default None
        Size of the smoothing window of the Savitsky-Golay filter.
        The default value ``None`` results in a window of two times the sampling rate (i.e. containing three values).
    smooth_poly_deg : int, default 2
        Degree of the polynomial used for fitting with the Savitsky-Golay filter.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    References
    ----------
    This Function is a generalization of the Spectrum based Spike flagging mechanism as presented in:

    [1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.

    Notes
    -----
    A value is flagged a spike, if:

    * The quotient to its preceding data point exceeds a certain bound:

      * :math:`|\\frac{x_k}{x_{k-1}}| > 1 +` ``raise_factor``, or
      * :math:`|\\frac{x_k}{x_{k-1}}| < 1 -` ``raise_factor``

    * The quotient of the second derivative :math:`x''`, at the preceding
      and subsequent timestamps is close enough to 1:

      * :math:`|\\frac{x''_{k-1}}{x''_{k+1}} | > 1 -` ``deriv_factor``, and
      * :math:`|\\frac{x''_{k-1}}{x''_{k+1}} | < 1 +` ``deriv_factor``

    * The dataset :math:`X = x_i, ..., x_{k-1}, x_{k+1}, ..., x_j`, with
      :math:`|t_{k-1} - t_i| = |t_j - t_{k+1}| =` ``noise_window`` fulfills the
      following condition:

      * ``noise_func``:math:`(X) <` ``noise_thresh``

    """

    dataseries, data_rate = retrieveTrustworthyOriginal(data, field, flagger)
    noise_func_map = {"covar": pd.Series.var, "rvar": pd.Series.std}
    noise_func = noise_func_map[noise_func.lower()]

    if smooth_window is None:
        smooth_window = 3 * pd.Timedelta(data_rate)
    else:
        smooth_window = pd.Timedelta(smooth_window)

    quotient_series = dataseries / dataseries.shift(+1)
    spikes = (quotient_series > (1 + raise_factor)) | (quotient_series < (1 - raise_factor))
    spikes = spikes[spikes == True]

    # loop through spikes: (loop may sound ugly - but since the number of spikes is supposed to not exceed the
    # thousands for year data - a loop going through all the spikes instances is much faster than
    # a rolling window, rolling all through a stacked year dataframe )

    # calculate some values, repeatedly needed in the course of the loop:

    filter_window_seconds = smooth_window.seconds
    smoothing_periods = int(np.ceil((filter_window_seconds / data_rate.n)))
    lower_dev_bound = 1 - deriv_factor
    upper_dev_bound = 1 + deriv_factor

    if smoothing_periods % 2 == 0:
        smoothing_periods += 1

    for spike in spikes.index:
        start_slice = spike - smooth_window
        end_slice = spike + smooth_window

        scnd_derivate = savgol_filter(
            dataseries[start_slice:end_slice], window_length=smoothing_periods, polyorder=smooth_poly_deg, deriv=2,
        )

        length = scnd_derivate.size
        test_ratio_1 = np.abs(scnd_derivate[int(((length + 1) / 2) - 2)] / scnd_derivate[int(((length + 1) / 2))])

        if lower_dev_bound < test_ratio_1 < upper_dev_bound:
            # apply noise condition:
            start_slice = spike - pd.Timedelta(noise_window)
            end_slice = spike + pd.Timedelta(noise_window)
            test_slice = dataseries[start_slice:end_slice].drop(spike)
            test_ratio_2 = np.abs(noise_func(test_slice) / test_slice.mean())
            # not a spike, we want to flag, if condition not satisfied:
            if test_ratio_2 > noise_thresh:
                spikes[spike] = False

        # not a spike, we want to flag, if condition not satisfied
        else:
            spikes[spike] = False

    spikes = spikes[spikes == True]

    flagger = flagger.setFlags(field, spikes.index, **kwargs)
    return data, flagger


@register(masking='field')
def spikes_flagGrubbs(data, field, flagger, winsz, alpha=0.05, min_periods=8, check_lagged=False, **kwargs):
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
    flagger : saqc.flagger.BaseFlagger
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
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    References
    ----------
    introduction to the grubbs test:

    [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers

    """

    data = data.copy()
    datcol = data[field]
    to_group = pd.DataFrame(data={"ts": datcol.index, "data": datcol})
    to_flag = pd.Series(False, index=datcol.index)
    if isinstance(winsz, int):
        # period number defined test intervals
        grouper_series = pd.Series(data=np.arange(0, datcol.shape[0]), index=datcol.index)
        grouper_series_lagged = grouper_series + (winsz / 2)
        grouper_series = grouper_series.transform(lambda x: int(np.floor(x / winsz)))
        grouper_series_lagged = grouper_series_lagged.transform(lambda x: int(np.floor(x / winsz)))
        partitions = to_group.groupby(grouper_series)
        partitions_lagged = to_group.groupby(grouper_series_lagged)
    else:
        # offset defined test intervals:
        partitions = to_group.groupby(pd.Grouper(freq=winsz))
    for _, partition in partitions:
        if partition.shape[0] > min_periods:
            detected = smirnov_grubbs.two_sided_test_indices(partition["data"].values, alpha=alpha)
            detected = partition["ts"].iloc[detected]
            to_flag[detected.index] = True

    if check_lagged & isinstance(winsz, int):
        to_flag_lagged = pd.Series(False, index=datcol.index)
        for _, partition in partitions_lagged:
            if partition.shape[0] > min_periods:
                detected = smirnov_grubbs.two_sided_test_indices(partition["data"].values, alpha=alpha)
                detected = partition["ts"].iloc[detected]
                to_flag_lagged[detected.index] = True
        to_flag = to_flag & to_flag_lagged

    flagger = flagger.setFlags(field, loc=to_flag, **kwargs)
    return data, flagger
