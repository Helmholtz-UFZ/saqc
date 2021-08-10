"""

"""
def flagMissing(field, nodata, flag):
    """
    The function flags all values indicating missing data.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    nodata : any, default np.nan
        A value that defines missing data.
    flag : float, default BAD
        flag to set.
    """
    pass


def flagIsolated(field, gap_window, group_window, flag):
    """
    The function flags arbitrary large groups of values, if they are surrounded by sufficiently
    large data gaps.
    
    A gap is a timespan containing either no data or data invalid only (usually `nan`) .
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    gap_window : str
        The minimum size of the gap before and after a group of valid values, making this group considered an
        isolated group. See condition (2) and (3)
    group_window : str
        The maximum temporal extension allowed for a group that is isolated by gaps of size 'gap_window',
        to be actually flagged as isolated group. See condition (1).
    flag : float, default BAD
        flag to set.
    
    Notes
    -----
    A series of values :math:`x_k,x_{k+1},...,x_{k+n}`, with associated timestamps :math:`t_k,t_{k+1},...,t_{k+n}`,
    is considered to be isolated, if:
    
    1. :math:`t_{k+1} - t_n <` `group_window`
    2. None of the :math:`x_j` with :math:`0 < t_k - t_j <` `gap_window`, is valid (preceeding gap).
    3. None of the :math:`x_j` with :math:`0 < t_j - t_(k+n) <` `gap_window`, is valid (succeding gap).
    
    See Also
    --------
    :py:func:`flagMissing`
    """
    pass


def flagJumps(field, thresh, winsz, min_periods, flag):
    """
    Flag datapoints, where the mean of the values significantly changes (where the value course "jumps").
    
    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.
    thresh : float
        The threshold, the mean of the values have to change by, to trigger flagging.
    winsz : str
        The temporal extension, of the rolling windows, the mean values that are to be compared,
        are obtained from.
    min_periods : int, default 1
        Minimum number of periods that have to be present in a window of size `winsz`, so that
        the mean value obtained from that window is regarded valid.
    flag : float, default BAD
        flag to set.
    """
    pass


def flagChangePoints(field, stat_func, thresh_func, bwd_window, min_periods_bwd, fwd_window, min_periods_fwd, closed, reduce_window, reduce_func, flag):
    """
    Flag datapoints, where the parametrization of the process, the data is assumed to generate by, significantly
    changes.
    
    The change points detection is based on a sliding window search.
    
    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.
    stat_func : Callable[numpy.array, numpy.array]
         A function that assigns a value to every twin window. Left window content will be passed to first variable,
        right window content will be passed to the second.
    thresh_func : Callable[numpy.array, numpy.array]
        A function that determines the value level, exceeding wich qualifies a timestamps stat func value as denoting a
        changepoint.
    bwd_window : str
        The left (backwards facing) windows temporal extension (freq-string).
    min_periods_bwd : {str, int}
        Minimum number of periods that have to be present in a backwards facing window, for a changepoint test to be
        performed.
    fwd_window : {None, str}, default None
        The right (forward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, str, int}, default None
        Minimum number of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.
    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.
    reduce_func : Callable[[numpy.ndarray, numpy.ndarray], int], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.
    flag : float, default BAD
        flag to set.
    
    Returns
    -------
    """
    pass


def assignChangePointCluster(field, stat_func, thresh_func, bwd_window, min_periods_bwd, fwd_window, min_periods_fwd, closed, reduce_window, reduce_func, flag_changepoints, model_by_resids, assign_cluster, flag):
    """
    Assigns label to the data, aiming to reflect continous regimes of the processes the data is assumed to be
    generated by.
    The regime change points detection is based on a sliding window search.
    
    Note, that the cluster labels will be stored to the `field` field of the input data, so that the data that is
    clustered gets overridden.
    
    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.
    stat_func : Callable[[numpy.array, numpy.array], float]
        A function that assigns a value to every twin window. Left window content will be passed to first variable,
        right window content will be passed to the second.
    thresh_func : Callable[numpy.array, numpy.array], float]
        A function that determines the value level, exceeding wich qualifies a timestamps stat func value as denoting a
        changepoint.
    bwd_window : str
        The left (backwards facing) windows temporal extension (freq-string).
    min_periods_bwd : int
        Minimum number of periods that have to be present in a backwards facing window, for a changepoint test to be
        performed.
    fwd_window : {None, str}, default None
        The right (forward facing) windows temporal extension (freq-string).
    min_periods_fwd : {None, int}, default None
        Minimum number of periods that have to be present in a forward facing window, for a changepoint test to be
        performed.
    closed : {'right', 'left', 'both', 'neither'}, default 'both'
        Determines the closure of the sliding windows.
    reduce_window : {None, str}, default None
        The sliding window search method is not an exact CP search method and usually there wont be
        detected a single changepoint, but a "region" of change around a changepoint.
        If `reduce_window` is given, for every window of size `reduce_window`, there
        will be selected the value with index `reduce_func(x, y)` and the others will be dropped.
        If `reduce_window` is None, the reduction window size equals the
        twin window size, the changepoints have been detected with.
    reduce_func : Callable[[numpy.array, numpy.array], numpy.array], default lambda x, y: x.argmax()
        A function that must return an index value upon input of two arrays x and y.
        First input parameter will hold the result from the stat_func evaluation for every
        reduction window. Second input parameter holds the result from the thresh_func evaluation.
        The default reduction function just selects the value that maximizes the stat_func.
    flag_changepoints : bool, default False
        If true, the points, where there is a change in data modelling regime detected gets flagged.
    model_by_resids : bool, default False
        If True, the data is replaced by the stat_funcs results instead of regime labels.
    assign_cluster : bool, default True
        Is set to False, if called by function that oly wants to calculate flags.
    flag : float, default BAD
        flag to set.
    
    Returns
    -------
    """
    pass


def flagConstants(field, thresh, window, flag):
    """
    This functions flags plateaus/series of constant values of length `window` if
    their maximum total change is smaller than thresh.
    
    Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:
    
    (1) n > `window`
    (2) |(y(t + i) - (t + j)| < `thresh`, for all i,j in [0, 1, ..., n]
    
    Flag values are (semi-)constant.
    
    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-flagged.
    thresh : float
        Upper bound for the maximum total change of an interval to be flagged constant.
    window : str
        Lower bound for the size of an interval to be flagged constant.
    flag : float, default BAD
        flag to set.
    """
    pass


def flagByVariance(field, window, thresh, max_missing, max_consec_missing, flag):
    """
    Function flags plateaus/series of constant values. Any interval of values y(t),..y(t+n) is flagged, if:
    
    (1) n > `window`
    (2) variance(y(t),...,y(t+n) < `thresh`
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    window : str
        Only intervals of minimum size "window" have the chance to get flagged as constant intervals
    thresh : float
        The upper bound, the variance of an interval must not exceed, if the interval wants to be flagged a plateau.
    max_missing : {None, int}, default None
        Maximum number of nan values tolerated in an interval, for retrieving a valid
        variance from it. (Intervals with a number of nans exceeding "max_missing"
        have no chance to get flagged a plateau!)
    max_consec_missing : {None, int}, default None
        Maximum number of consecutive nan values allowed in an interval to retrieve a
        valid  variance from it. (Intervals with a number of nans exceeding
        "max_consec_missing" have no chance to get flagged a plateau!)
    flag : float, default BAD
        flag to set.
    """
    pass


def fitPolynomial(field, winsz, polydeg, numba, eval_flags, min_periods, return_residues, flag):
    """
    Function fits a polynomial model to the data and returns the fitted data curve.
    
    The fit is calculated by fitting a polynomial of degree `polydeg` to a data slice
    of size `winsz`, that has x at its center.
    
    Note, that the resulting fit is stored to the `field` field of the input data, so that the original data, the
    polynomial is fitted to, gets overridden.
    
    Note, that, if data[field] is not alligned to an equidistant frequency grid, the window size passed,
    has to be an offset string. Also numba boost options don`t apply for irregularly sampled
    timeseries.
    
    Note, that calculating the residues tends to be quite costy, because a function fitting is perfomed for every
    sample. To improve performance, consider the following possibillities:
    
    In case your data is sampled at an equidistant frequency grid:
    
    (1) If you know your data to have no significant number of missing values, or if you do not want to
        calculate residues for windows containing missing values any way, performance can be increased by setting
        min_periods=winsz.
    
    (2) If your data consists of more then around 200000 samples, setting numba=True, will boost the
        calculations up to a factor of 5 (for samplesize > 300000) - however for lower sample sizes,
        numba will slow down the calculations, also, up to a factor of 5, for sample_size < 50000.
        By default (numba='auto'), numba is set to true, if the data sample size exceeds 200000.
    
    in case your data is not sampled at an equidistant frequency grid:
    
    (1) Harmonization/resampling of your data will have a noticable impact on polyfittings performance - since
        numba_boost doesnt apply for irregularly sampled data in the current implementation.
    
    Note, that in the current implementation, the initial and final winsz/2 values do not get fitted.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
    winsz : {str, int}
        The size of the window you want to use for fitting. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension. The window will be centered around the vaule-to-be-fitted.
        For regularly sampled timeseries the period number will be casted down to an odd number if
        even.
    polydeg : int
        The degree of the polynomial used for fitting
    numba : {True, False, "auto"}, default "auto"
        Wheather or not to apply numbas just-in-time compilation onto the poly fit function. This will noticably
        increase the speed of calculation, if the sample size is sufficiently high.
        If "auto" is selected, numba compatible fit functions get applied for data consisiting of > 200000 samples.
    eval_flags : bool, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
    min_periods : {int, None}, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.
    return_residues : bool, default False
        Internal parameter. Makes the method return the residues instead of the fit.
    flag : float, default BAD
        flag to set.
    """
    pass


def flagDriftFromNorm(field, fields, segment_freq, norm_spread, norm_frac, metric, linkage_method, flag):
    """
    The function flags value courses that significantly deviate from a group of normal value courses.
    
    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".
    
    See the Notes section for a more detailed presentation of the algorithm
    
    Parameters
    ----------
    field : str
        A dummy parameter.
    fields : str
        List of fieldnames in data, determining which variables are to be included into the flagging process.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    norm_spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[[numpy.array, numpy.array], float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    flag : float, default BAD
        flag to set.
    
    Notes
    -----
    following steps are performed for every data "segment" of length `segment_freq` in order to find the
    "abnormal" data:
    
    1. Calculate the distances :math:`d(x_i,x_j)` for all :math:`x_i` in parameter `fields`. (with :math:`d`
       denoting the distance function
       passed to the parameter `metric`.
    2. Calculate a dendogram with a hierarchical linkage algorithm, specified by the parameter `linkage_method`.
    3. Flatten the dendogram at the level, the agglomeration costs exceed the value given by the parameter `norm_spread`
    4. check if there is a cluster containing more than `norm_frac` percentage of the variables in fields.
    
        1. if yes: flag all the variables that are not in that cluster (inside the segment)
        2. if no: flag nothing
    
    The main parameter giving control over the algorithms behavior is the `norm_spread` parameter, that determines
    the maximum spread of a normal group by limiting the costs, a cluster agglomeration must not exceed in every
    linkage step.
    For singleton clusters, that costs just equal half the distance, the timeseries in the clusters, have to
    each other. So, no timeseries can be clustered together, that are more then
    2*`norm_spread` distanted from each other.
    When timeseries get clustered together, this new clusters distance to all the other timeseries/clusters is
    calculated according to the linkage method specified by `linkage_method`. By default, it is the minimum distance,
    the members of the clusters have to each other.
    Having that in mind, it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).
    
    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """
    pass


def flagDriftFromReference(field, fields, segment_freq, thresh, metric, flag):
    """
    The function flags value courses that deviate from a reference course by a margin exceeding a certain threshold.
    
    The deviation is measured by the distance function passed to parameter metric.
    
    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.
    fields : str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    thresh : float
        The threshod by wich normal variables can deviate from the reference variable at max.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    flag : float, default BAD
        flag to set.
    
    Notes
    -----
    it is advisable to choose a distance function, that can be well interpreted in the units
    dimension of the measurement and where the interpretation is invariant over the length of the timeseries.
    That is, why, the "averaged manhatten metric" is set as the metric default, since it corresponds to the
    averaged value distance, two timeseries have (as opposed by euclidean, for example).
    """
    pass


def flagDriftFromScaledNorm(field, fields_scale1, fields_scale2, segment_freq, norm_spread, norm_frac, metric, linkage_method, flag):
    """
    The function linearly rescales one set of variables to another set of variables with a different scale and then
    flags value courses that significantly deviate from a group of normal value courses.
    
    The two sets of variables can be linearly scaled one to another and hence the scaling transformation is performed
    via linear regression: A linear regression is performed on each pair of variables giving a slope and an intercept.
    The transformation is then calculated a the median of all the calculated slopes and intercepts.
    
    Once the transformation is performed, the function flags those values, that deviate from a group of normal values.
    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `norm_frac` percent of the
    variables in "fields".
    
    Parameters
    ----------
    field : str
        A dummy parameter.
    fields_scale1 : str
        List of fieldnames in data to be included into the flagging process which are scaled according to scaling
        scheme 1.
    fields_scale2 : str
        List of fieldnames in data to be included into the flagging process which are scaled according to scaling
        scheme 2.
    segment_freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    norm_spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    norm_frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    linkage_method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    Documentation of the underlying hierarchical clustering algorithm:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Introduction to Hierarchical clustering:
        [2] https://en.wikipedia.org/wiki/Hierarchical_clustering
    """
    pass


def correctDrift(field, maint_data_field, driftModel, cal_mean, flag_maint_period, flag):
    """
    The function corrects drifting behavior.
    
    See the Notes section for an overview over the correction algorithm.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to correct.
    maint_data_field : str
        The fieldname of the datacolumn holding the support-points information.
        The maint data is to expected to have following form:
        The series' timestamp itself represents the beginning of a
        maintenance event, wheras the values represent the endings of the maintenance intervals.
    driftModel : Callable
        A modelfunction describing the drift behavior, that is to be corrected.
        The model function must always contain the keyword parameters 'origin' and 'target'.
        The starting parameter must always be the parameter, by wich the data is passed to the model.
        After the data parameter, there can occure an arbitrary number of model calibration arguments in
        the signature.
        See the Notes section for an extensive description.
    cal_mean : int, default 5
        The number of values the mean is computed over, for obtaining the value level directly after and
        directly before maintenance event. This values are needed for shift calibration. (see above description)
    flag_maint_period : bool, default False
        Whether or not to flag the values obtained while maintenance.
    flag : float, default BAD
        flag to set.
    
    Notes
    -----
    It is assumed, that between support points, there is a drift effect shifting the meassurements in a way, that
    can be described, by a model function M(t, *p, origin, target). (With 0<=t<=1, p being a parameter set, and origin,
    target being floats).
    
    Note, that its possible for the model to have no free parameters p at all. (linear drift mainly)
    
    The drift model, directly after the last support point (t=0),
    should evaluate to the origin - calibration level (origin), and directly before the next support point
    (t=1), it should evaluate to the target calibration level (target).
    
    M(0, *p, origin, target) = origin
    M(1, *p, origin, target) = target
    
    The model is than fitted to any data chunk in between support points, by optimizing the parameters p*, and
    thus, obtaining optimal parameterset P*.
    
    The new values at t are computed via:
    
    new_vals(t) = old_vals(t) + M(t, *P, origin, target) - M_drift(t, *P, origin, new_target)
    
    Wheras new_target represents the value level immediately after the nex support point.
    
    Examples
    --------
    Some examples of meaningful driftmodels.
    
    Linear drift modell (no free parameters).
    
    >>> M = lambda t, origin, target: origin + t*target
    
    exponential drift model (exponential raise!)
    
    >>> expFunc = lambda t, a, b, c: a + b * (np.exp(c * x) - 1)
    >>> M = lambda t, p, origin, target: expFunc(t, (target - origin) / (np.exp(abs(c)) - 1), abs(c))
    
    Exponential and linear driftmodels are part of the ts_operators library, under the names
    expDriftModel and linearDriftModel.
    """
    pass


def correctRegimeAnomaly(field, cluster_field, model, regime_transmission, x_date):
    """
    Function fits the passed model to the different regimes in data[field] and tries to correct
    those values, that have assigned a negative label by data[cluster_field].
    
    Currently, the only correction mode supported is the "parameter propagation."
    
    This means, any regime :math:`z`, labeled negatively and being modeled by the parameters p, gets corrected via:
    
    :math:`z_{correct} = z + (m(p^*) - m(p))`,
    
    where :math:`p^*` denotes the parameter set belonging to the fit of the nearest not-negatively labeled cluster.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to correct.
    cluster_field : str
        A string denoting the field in data, holding the cluster label for the data you want to correct.
    model : Callable
        The model function to be fitted to the regimes.
        It must be a function of the form :math:`f(x, *p)`, where :math:`x` is the ``numpy.array`` holding the
        independent variables and :math:`p` are the model parameters that are to be obtained by fitting.
        Depending on the `x_date` parameter, independent variable x will either be the timestamps
        of every regime transformed to seconds from epoch, or it will be just seconds, counting the regimes length.
    regime_transmission : {None, str}, default None:
        If an offset string is passed, a data chunk of length `regime_transimission` right at the
        start and right at the end is ignored when fitting the model. This is to account for the
        unreliability of data near the changepoints of regimes.
    x_date : bool, default False
        If True, use "seconds from epoch" as x input to the model func, instead of "seconds from regime start".
    """
    pass


def correctOffset():
    """
    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to correct.
    flags : saqc.Flags
        Container to store flags of the data.
    max_mean_jump : float
        when searching for changepoints in mean - this is the threshold a mean difference in the
        sliding window search must exceed to trigger changepoint detection.
    normal_spread : float
        threshold denoting the maximum, regimes are allowed to abolutely differ in their means
        to form the "normal group" of values.
    search_winsz : str
        Size of the adjacent windows that are used to search for the mean changepoints.
    min_periods : int
        Minimum number of periods a search window has to contain, for the result of the changepoint
        detection to be considered valid.
    regime_transmission : {None, str}, default None:
        If an offset string is passed, a data chunk of length `regime_transimission` right from the
        start and right before the end of any regime is ignored when calculating a regimes mean for data correcture.
        This is to account for the unrelyability of data near the changepoints of regimes.
    """
    pass


def flagRegimeAnomaly(field, cluster_field, norm_spread, linkage_method, metric, norm_frac, flag):
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
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
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
    flag : float, default BAD
        flag to set.
    """
    pass


def assignRegimeAnomaly(field, cluster_field, norm_spread, linkage_method, metric, norm_frac, set_cluster, set_flags, flag):
    """
    A function to detect values belonging to an anomalous regime regarding modelling regimes of field.
    
    The function changes the value of the regime cluster labels to be negative.
    
    "Normality" is determined in terms of a maximum spreading distance, regimes must not exceed in respect
    to a certain metric and linkage method.
    
    In addition, only a range of regimes is considered "normal", if it models more then `norm_frac` percentage of
    the valid samples in "field".
    
    Note, that you must detect the regime changepoints prior to calling this function. (They are expected to be stored
    parameter `cluster_field`.)
    
    Note, that it is possible to perform hypothesis tests for regime equality by passing the metric
    a function for p-value calculation and selecting linkage method "complete".
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
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
    set_cluster : bool, default False
        If True, all data, considered "anormal", gets assigned a negative clusterlabel. This option
        is present for further use (correction) of the anomaly information.
    set_flags : bool, default True
        Wheather or not to flag abnormal values (do not flag them, if you want to correct them
        afterwards, becasue flagged values usually are not visible in further tests.).
    flag : float, default BAD
        flag to set.
    """
    pass


def forceFlags(field, flag, kwargs):
    """
    Set whole column to a flag value.
    
    Parameters
    ----------
    field : str
        columns name that holds the data
    flag : float, default BAD
        flag to set
    kwargs : dict
        unused
    
    See Also
    --------
    clearFlags : set whole column to UNFLAGGED
    flagUnflagged : set flag value at all unflagged positions
    """
    pass


def clearFlags(field, kwargs):
    """
    Set whole column to UNFLAGGED.
    
    Parameters
    ----------
    field : str
        columns name that holds the data
    kwargs : dict
        unused
    
    See Also
    --------
    forceFlags : set whole column to a flag value
    flagUnflagged : set flag value at all unflagged positions
    """
    pass


def flagUnflagged(field, flag, kwargs):
    """
    Function sets a flag at all unflagged positions.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flag : float, default BAD
        flag value to set
    kwargs : Dict
        unused
    
    See Also
    --------
    clearFlags : set whole column to UNFLAGGED
    forceFlags : set whole column to a flag value
    """
    pass


def flagManual(field, mdata, mflag, method, flag):
    """
    Flag data by given, "manually generated" data.
    
    The data is flagged at locations where `mdata` is equal to a provided flag (`mflag`).
    The format of mdata can be an indexed object, like pd.Series, pd.Dataframe or dios.DictOfSeries,
    but also can be a plain list- or array-like.
    How indexed mdata is aligned to data is specified via the `method` parameter.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    mdata : {pd.Series, pd.Dataframe, DictOfSeries}
        The "manually generated" data
    mflag : scalar
        The flag that indicates data points in `mdata`, of wich the projection in data should be flagged.
    
    method : {'plain', 'ontime', 'left-open', 'right-open'}, default plain
        Defines how mdata is projected on data. Except for the 'plain' method, the methods assume mdata to have an
        index.
    
        * 'plain': mdata must have the same length as data and is projected one-to-one on data.
        * 'ontime': works only with indexed mdata. mdata entries are matched with data entries that have the same index.
        * 'right-open': mdata defines intervals, values are to be projected on.
          The intervals are defined by any two consecutive timestamps t_1 and 1_2 in mdata.
          the value at t_1 gets projected onto all data timestamps t with t_1 <= t < t_2.
        * 'left-open': like 'right-open', but the projected interval now covers all t with t_1 < t <= t_2.
    
    flag : float, default BAD
        flag to set.
    
    Examples
    --------
    An example for mdata
    >>> mdata = pd.Series([1,0,1], index=pd.to_datetime(['2000-02', '2000-03', '2001-05']))
    >>> mdata
    2000-02-01    1
    2000-03-01    0
    2001-05-01    1
    dtype: int64
    
    On *dayly* data, with the 'ontime' method, only the provided timestamnps are used.
    Bear in mind that only exact timestamps apply, any offset will result in ignoring
    the timestamp.
    >>> _, fl = flagManual(data, field, flags, mdata, mflag=1, method='ontime')
    >>> fl[field] > UNFLAGGED
    2000-01-31    False
    2000-02-01    True
    2000-02-02    False
    2000-02-03    False
    ..            ..
    2000-02-29    False
    2000-03-01    True
    2000-03-02    False
    Freq: D, dtype: bool
    
    With the 'right-open' method, the mdata is forward fill:
    >>> _, fl = flagManual(data, field, flags, mdata, mflag=1, method='right-open')
    >>> fl[field] > UNFLAGGED
    2000-01-31    False
    2000-02-01    True
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    False
    2000-03-02    False
    Freq: D, dtype: bool
    
    With the 'left-open' method, backward filling is used:
    >>> _, fl = flagManual(data, field, flags, mdata, mflag=1, method='left-open')
    >>> fl[field] > UNFLAGGED
    2000-01-31    False
    2000-02-01    False
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    True
    2000-03-02    False
    Freq: D, dtype: bool
    """
    pass


def flagDummy(field):
    """
    Function does nothing but returning data and flags.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass


def process(field, func, nodata):
    """
    generate/process data with generically defined functions.
    
    The functions can depend on on any of the fields present in data.
    
    Formally, what the function does, is the following:
    
    1.  Let F be a Callable, depending on fields f_1, f_2,...f_K, (F = F(f_1, f_2,...f_K))
        Than, for every timestamp t_i that occurs in at least one of the timeseries data[f_j] (outer join),
        The value v_i is computed via:
        v_i = data([f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]), if all data[f_j][t_i] do exist
        v_i = `nodata`, if at least one of the data[f_j][t_i] is missing.
    2.  The result is stored to data[field] (gets generated if not present)
    
    Parameters
    ----------
    field : str
        The fieldname of the column, where you want the result from the generic expressions processing to be written to.
    func : Callable
        The data processing function with parameter names that will be
        interpreted as data column entries.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data
    
    Examples
    --------
    Some examples on what to pass to the func parameter:
    To compute the sum of the variables "temperature" and "uncertainty", you would pass the function:
    
    >>> lambda temperature, uncertainty: temperature + uncertainty
    
    You also can pass numpy and pandas functions:
    
    >>> lambda temperature, uncertainty: np.round(temperature) * np.sqrt(uncertainty)
    """
    pass


def flag(field, func, nodata, flag):
    """
    a function to flag a data column by evaluation of a generic expression.
    
    The expression can depend on any of the fields present in data.
    
    Formally, what the function does, is the following:
    
    Let X be an expression, depending on fields f_1, f_2,...f_K, (X = X(f_1, f_2,...f_K))
    Than for every timestamp t_i in data[field]:
    data[field][t_i] is flagged if X(data[f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]) is True.
    
    Note, that all value series included in the expression to evaluate must be labeled identically to field.
    
    Note, that the expression is passed in the form of a Callable and that this callables variable names are
    interpreted as actual names in the data header. See the examples section to get an idea.
    
    Note, that all the numpy functions are available within the generic expressions.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, where you want the result from the generic expressions evaluation to be projected
        to.
    func : Callable
        The expression that is to be evaluated is passed in form of a callable, with parameter names that will be
        interpreted as data column entries. The Callable must return an boolen array like.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data
    flag : float, default BAD
        flag to set.
    
    Examples
    --------
    Some examples on what to pass to the func parameter:
    To flag the variable `field`, if the sum of the variables
    "temperature" and "uncertainty" is below zero, you would pass the function:
    
    >>> lambda temperature, uncertainty: temperature + uncertainty < 0
    
    There is the reserved name 'This', that always refers to `field`. So, to flag field if field is negative, you can
    also pass:
    
    >>> lambda this: this < 0
    
    If you want to make dependent the flagging from flags already present in the data, you can use the built-in
    ``isflagged`` method. For example, to flag the 'temperature', if 'level' is flagged, you would use:
    
    >>> lambda level: isflagged(level)
    
    You can furthermore specify a flagging level, you want to compare the flags to. For example, for flagging
    'temperature', if 'level' is flagged at a level named DOUBTFUL or worse, use:
    
    >>> lambda level: isflagged(level, flag=DOUBTFUL, comparator='>')
    
    If you are unsure about the used flaggers flagging level names, you can use the reserved key words BAD, UNFLAGGED
    and GOOD, to refer to the worst (BAD), best(GOOD) or unflagged (UNFLAGGED) flagging levels. For example.
    
    >>> lambda level: isflagged(level, flag=UNFLAGGED, comparator='==')
    
    Your expression also is allowed to include pandas and numpy functions
    
    >>> lambda level: np.sqrt(level) > 7
    """
    pass


def interpolateByRolling(field, winsz, func, center, min_periods, flag):
    """
    Interpolates nan-values in the data by assigning them the aggregation result of the window surrounding them.
    
    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-interpolated.
    
    winsz : int, str
        The size of the window, the aggregation is computed from. An integer define the number of periods to be used,
        an string is interpreted as an offset. ( see `pandas.rolling` for more information).
        Integer windows may result in screwed aggregations if called on none-harmonized or irregular data.
    
    func : Callable
        The function used for aggregation.
    
    center : bool, default True
        Center the window around the value. Can only be used with integer windows, otherwise it is silently ignored.
    
    min_periods : int
        Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
        computed.
    
    flag : float or None, default UNFLAGGED
        Flag that is to be inserted for the interpolated values. If ``None`` no flags are set.
    """
    pass


def interpolateInvalid(field, method, inter_order, inter_limit, flag, downgrade_interpolation):
    """
    Function to interpolate nan values in the data.
    
    There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
    the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.
    
    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-interpolated.
    
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
        The interpolation method to use.
    
    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    
    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `inter_limit` successive nan entries.
    
    flag : float or None, default UNFLAGGED
        Flag that is set for interpolated values. If ``None``, no flags are set at all.
    
    downgrade_interpolation : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``inter_order``, or
        simply because not enough values are present in a interval.
    """
    pass


def interpolateIndex(field, freq, method, inter_order, inter_limit, downgrade_interpolation):
    """
    Function to interpolate the data at regular (equidistant) timestamps (or Grid points).
    
    Note, that the interpolation will only be calculated, for grid timestamps that have a preceding AND a succeeding
    valid data value within "freq" range.
    
    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-interpolated.
    
    freq : str
        An Offset String, interpreted as the frequency of
        the grid you want to interpolate your data at.
    
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    
    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    
    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `inter_limit` successive nan entries.
    
    downgrade_interpolation : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``inter_order``, or
        simply because not enough values are present in a interval.
    
    """
    pass


def flagByStatLowPass(field):
    """
    Flag *chunks* of length, `winsz`:
    
    1. If they excexceed `thresh` with regard to `stat`:
    2. If all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_winsz`,
       `excexceed `sub_thresh` with regard to `stat`:
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass


def flagByStray(field, partition_freq, partition_min, iter_start, alpha, flag):
    """
    Flag outliers in 1-dimensional (score) data with the STRAY Algorithm.
    
    Find more information on the algorithm in References [1].
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
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
    
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2019). Anomaly detection in high dimensional data.
        arXiv preprint arXiv:1908.04000.
    """
    pass


def flagMVScores(field, fields, trafo, alpha, n_neighbors, scoring_func, iter_start, stray_partition, stray_partition_min, trafo_on_partition, reduction_range, reduction_drop_flagged, reduction_thresh, reduction_min_periods, flag):
    """
    The algorithm implements a 3-step outlier detection procedure for simultaneously flagging of higher dimensional
    data (dimensions > 3).
    
    In references [1], the procedure is introduced and exemplified with an application on hydrological data.
    
    See the notes section for an overview over the algorithms basic steps.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
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
    flag : float, default BAD
        flag to set.
    
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
    pass


def flagRaise(field, thresh, raise_window, intended_freq, average_window, mean_raise_factor, min_slope, min_slope_weight, numba_boost, flag):
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
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
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
        deprecated ?
    flag : float, default BAD
        flag to set.
    
    Notes
    -----
    The value :math:`x_{k}` of a time series :math:`x` with associated
    timestamps :math:`t_i`, is flagged a raise, if:
    
    * There is any value :math:`x_{s}`, preceeding :math:`x_{k}` within `raise_window` range, so that:
    
      * :math:`M = |x_k - x_s | >`  `thresh` :math:`> 0`
    
    * The weighted average :math:`\mu^{*}` of the values, preceding :math:`x_{k}` within `average_window`
      range indicates, that :math:`x_{k}` does not return from an "outlierish" value course, meaning that:
    
      * :math:`x_k > \mu^* + ( M` / `mean_raise_factor` :math:`)`
    
    * Additionally, if `min_slope` is not `None`, :math:`x_{k}` is checked for being sufficiently divergent from its
      very predecessor :max:`x_{k-1}`$, meaning that, it is additionally checked if:
    
      * :math:`x_k - x_{k-1} >` `min_slope`
      * :math:`t_k - t_{k-1} >` `min_slope_weight` :math:`\times` `intended_freq`
    """
    pass


def flagMAD(field, window, flag):
    """
    The function represents an implementation of the modyfied Z-score outlier detection method.
    
    See references [1] for more details on the algorithm.
    
    Note, that the test needs the input data to be sampled regularly (fixed sampling rate).
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
    window : str
       Offset string. Denoting the windows size that the "Z-scored" values have to lie in.
    z: float, default 3.5
        The value the Z-score is tested against. Defaulting to 3.5 (Recommendation of [1])
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    pass


def flagOffset(field, thresh, tolerance, window, rel_thresh, numba_kickin, flag):
    """
    A basic outlier test that is designed to work for harmonized and not harmonized data.
    
    The test classifies values/value courses as outliers by detecting not only a rise in value, but also,
    checking for a return to the initial value level.
    
    Values :math:`x_n, x_{n+1}, .... , x_{n+k}` of a timeseries :math:`x` with associated timestamps
    :math:`t_n, t_{n+1}, .... , t_{n+k}` are considered spikes, if
    
    1. :math:`|x_{n-1} - x_{n + s}| >` `thresh`, for all :math:`s \in [0,1,2,...,k]`
    
    2. :math:`|x_{n-1} - x_{n+k+1}| <` `tolerance`
    
    3. :math:`|t_{n-1} - t_{n+k+1}| <` `window`
    
    Note, that this definition of a "spike" not only includes one-value outliers, but also plateau-ish value courses.
    
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged. (Here a dummy, for structural reasons)
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
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    The implementation is a time-window based version of an outlier test from the UFZ Python library,
    that can be found here:
    
    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py
    """
    pass


def flagByGrubbs(field, winsz, alpha, min_periods, flag):
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
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
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
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    introduction to the grubbs test:
    
    [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
    """
    pass


def flagRange(field, min, max, flag):
    """
    Function flags values not covered by the closed interval [`min`, `max`].
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    min : float
        Lower bound for valid data.
    max : float
        Upper bound for valid data.
    flag : float, default BAD
        flag to set.
    """
    pass


def flagCrossStatistic(field, fields, thresh, cross_stat, flag):
    """
    Function checks for outliers relatively to the "horizontal" input data axis.
    
    For `fields` :math:`=[f_1,f_2,...,f_N]` and timestamps :math:`[t_1,t_2,...,t_K]`, the following steps are taken
    for outlier detection:
    
    1. All timestamps :math:`t_i`, where there is one :math:`f_k`, with :math:`data[f_K]` having no entry at
       :math:`t_i`, are excluded from the following process (inner join of the :math:`f_i` fields.)
    2. for every :math:`0 <= i <= K`, the value
       :math:`m_j = median(\{data[f_1][t_i], data[f_2][t_i], ..., data[f_N][t_i]\})` is calculated
    2. for every :math:`0 <= i <= K`, the set
       :math:`\{data[f_1][t_i] - m_j, data[f_2][t_i] - m_j, ..., data[f_N][t_i] - m_j\}` is tested for outliers with the
       specified method (`cross_stat` parameter).
    
    Parameters
    ----------
    field : str
        A dummy parameter.
    fields : str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    thresh : float
        Threshold which the outlier score of an value must exceed, for being flagged an outlier.
    cross_stat : {'modZscore', 'Zscore'}, default 'modZscore'
        Method used for calculating the outlier scores.
    
        * ``'modZscore'``: Median based "sigma"-ish approach. See Referenecs [1].
        * ``'Zscore'``: Score values by how many times the standard deviation they differ from the median.
          See References [1]
    
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    pass


def flagPatternByDTW(field, flag):
    """
    Pattern recognition via wavelets.
    
    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the wavelet algorithm as presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged
    
    Parameters
    ----------
    
    field : str
        The fieldname of the data column, you want to correct.
    flag : float, default BAD
        flag to set.
    
    kwargs
    
    References
    ----------
    
    The underlying pattern recognition algorithm using wavelets is documented here:
    [1] Maharaj, E.A. (2002): Pattern Recognition of Time Series using Wavelets. In: Härdle W., Rönz B. (eds) Compstat. Physica, Heidelberg, 978-3-7908-1517-7.
    
    The documentation of the python package used for the wavelt decomposition can be found here:
    [2] https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families
    """
    pass


def flagPatternByWavelet(field, flag):
    """
    Pattern Recognition via Dynamic Time Warping.
    
    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the dynamic time warping algorithm as presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged
    
    Parameters
    ----------
    
    field : str
        The fieldname of the data column, you want to correct.
    flag : float, default BAD
        flag to set.
    
    References
    ----------
    Find a nice description of underlying the Dynamic Time Warping Algorithm here:
    
    [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
    """
    pass


def aggregate(field, freq, value_func, flag_func, method, flag):
    """
    A method to "regularize" data by aggregating (resampling) data at a regular timestamp.
    
    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).
    
    The data will therefor get aggregated with a function, specified by the `value_func` parameter and
    the result gets projected onto the new timestamps with a method, specified by "method".
    
    The following method (keywords) are available:
    
    * ``'nagg'``: (aggreagtion to nearest) - all values in the range (+/- freq/2) of a grid point get aggregated with
      `agg_func`. and assigned to it. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'bagg'``: (backwards aggregation) - all values in a sampling interval get aggregated with agg_func and the
      result gets assigned to the last regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'fagg'``: (forward aggregation) - all values in a sampling interval get aggregated with agg_func and the result
      gets assigned to the next regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.
    
    Note, that, if there is no valid data (exisitng and not-na) available in a sampling interval assigned to a regular
    timestamp by the selected method, nan gets assigned to this timestamp. The associated flag will be of value
    ``UNFLAGGED``.
    
    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    
    freq : str
        The sampling frequency the data is to be aggregated (resampled) at.
    
    value_func : Callable
        The function you want to use for aggregation.
    
    flag_func : Callable
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).
    
    method : {'fagg', 'bagg', 'nagg'}, default 'nagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceeding, succeeding or
        "surrounding" interval). See description above for more details.
    
    flag : float, default BAD
        flag to set.
    
    """
    pass


def linear(field, freq):
    """
    A method to "regularize" data by interpolating linearly the data at regular timestamp.
    
    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).
    
    Interpolated values will get assigned the worst flag within freq-range.
    
    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.
    
    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``UNFLAGGED``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    """
    pass


def interpolate(field, freq, method, order):
    """
    A method to "regularize" data by interpolating the data at regular timestamp.
    
    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).
    
    Interpolated values will get assigned the worst flag within freq-range.
    
    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.
    
    Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
    and NOT ``'linear'``.
    
    Note: the `method` will likely and significantly alter values and shape of ``data[field]``. The original data is
    kept in the data dios and assigned to the fieldname ``field + '_original'``.
    
    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``UNFLAGGED``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
        The interpolation method you want to apply.
    
    order : int, default 1
        If your selected interpolation method can be performed at different *orders* - here you pass the desired
        order.
    """
    pass


def mapToOriginal(field, method):
    """
    The Function function "undoes" regularization, by regaining the original data and projecting the
    flags calculated for the regularized data onto the original ones.
    
    Afterwards the regularized data is removed from the data dios and ``'field'`` will be associated
    with the original data "again".
    
    Wherever the flags in the original data are "better" then the regularized flags projected on them,
    they get overridden with this regularized flags value.
    
    Which regularized flags are to be projected on which original flags, is controlled by the "method" parameters.
    
    Generally, if you regularized with the method "X", you should pass the method "inverse_X" to the deharmonization.
    If you regularized with an interpolation, the method "inverse_interpolation" would be the appropriate choice.
    Also you should pass the same drop flags keyword.
    
    The deharm methods in detail:
    ("original_flags" are associated with the original data that is to be regained,
    "regularized_flags" are associated with the regularized data that is to be "deharmonized",
    "freq" refers to the regularized datas sampling frequencie)
    
    * ``'inverse_nagg'``: all original_flags within the range *+/- freq/2* of a regularized_flag, get assigned this
      regularized flags value. (if regularized_flags > original_flag)
    * ``'inverse_bagg'``: all original_flags succeeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_fagg'``: all original_flags preceeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)
    
    * ``'inverse_interpolation'``: all original_flags within the range *+/- freq* of a regularized_flag, get assigned this
      regularized flags value (if regularized_flag > original_flag).
    
    * ``'inverse_nshift'``: That original_flag within the range +/- *freq/2*, that is nearest to a regularized_flag,
      gets the regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_bshift'``: That original_flag succeeding a source flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_nshift'``: That original_flag preceeding a regularized flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if source_flag > original_flag)
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-deharmonized.
    
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
            'inverse_interpolation'}
        The method used for projection of regularized flags onto original flags. See description above for more
        details.
    """
    pass


def shift(field, freq, method, freq_check):
    """
    Function to shift data and flags to a regular (equidistant) timestamp grid, according to ``method``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-shifted.
    
    freq : str
        An frequency Offset String that will be interpreted as the sampling rate you want the data to be shifted to.
    
    method : {'fshift', 'bshift', 'nshift'}, default 'nshift'
        Specifies how misaligned data-points get propagated to a grid timestamp.
        Following choices are available:
    
        * 'nshift' : every grid point gets assigned the nearest value in its range. (range = +/- 0.5 * `freq`)
        * 'bshift' : every grid point gets assigned its first succeeding value, if one is available in
          the succeeding sampling interval.
        * 'fshift' : every grid point gets assigned its ultimately preceding value, if one is available in
          the preceeding sampling interval.
    
    freq_check : {None, 'check', 'auto'}, default None
    
        * ``None`` : do not validate frequency-string passed to `freq`
        * 'check' : estimate frequency and log a warning if estimate miss matches frequency string passed to `freq`,
          or if no uniform sampling rate could be estimated
        * 'auto' : estimate frequency and use estimate. (Ignores `freq` parameter.)
    """
    pass


def resample(field, freq, agg_func, max_invalid_total_d, max_invalid_consec_d, max_invalid_total_f, max_invalid_consec_f, flag_agg_func, freq_check):
    """
    Function to resample the data. Afterwards the data will be sampled at regular (equidistant) timestamps
    (or Grid points). Sampling intervals therefor get aggregated with a function, specifyed by 'agg_func' parameter and
    the result gets projected onto the new timestamps with a method, specified by "method". The following method
    (keywords) are available:
    
    * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get aggregated with agg_func and assigned to it.
    * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the last
      grid point.
    * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the next
      grid point.
    
    
    Note, that. if possible, functions passed to agg_func will get projected internally onto pandas.resample methods,
    wich results in some reasonable performance boost - however, for this to work, you should pass functions that have
    the __name__ attribute initialised and the according methods name assigned to it.
    Furthermore, you shouldnt pass numpys nan-functions
    (``nansum``, ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and they will thus not
    trigger ``resample.func()``, but the slower ``resample.apply(nanfunc)``. Also, internally, no nans get passed to
    the functions anyway, so that there is no point in passing the nan functions.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-resampled.
    
    freq : str
        An Offset String, that will be interpreted as the frequency you want to resample your data with.
    
    agg_func : Callable
        The function you want to use for aggregation.
    
    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceding, succeeding or
        "surrounding" interval). See description above for more details.
    
    max_invalid_total_d : {None, int}, default None
        Maximum number of invalid (nan) datapoints, allowed per resampling interval. If max_invalid_total_d is
        exceeded, the interval gets resampled to nan. By default (``np.inf``), there is no bound to the number of nan
        values in an interval and only intervals containing ONLY nan values or those, containing no values at all,
        get projected onto nan
    
    max_invalid_consec_d : {None, int}, default None
        Maximum number of consecutive invalid (nan) data points, allowed per resampling interval.
        If max_invalid_consec_d is exceeded, the interval gets resampled to nan. By default (np.inf),
        there is no bound to the number of consecutive nan values in an interval and only intervals
        containing ONLY nan values, or those containing no values at all, get projected onto nan.
    
    max_invalid_total_f : {None, int}, default None
        Same as `max_invalid_total_d`, only applying for the flags. The flag regarded as "invalid" value,
        is the one passed to empty_intervals_flag (default=``BAD``).
        Also this is the flag assigned to invalid/empty intervals.
    
    max_invalid_consec_f : {None, int}, default None
        Same as `max_invalid_total_f`, only applying onto flags. The flag regarded as "invalid" value, is the one passed
        to empty_intervals_flag. Also this is the flag assigned to invalid/empty intervals.
    
    flag_agg_func : Callable, default: max
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).
    
    freq_check : {None, 'check', 'auto'}, default None
    
        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
          if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)
    """
    pass


def reindexFlags(field, method, source, freq):
    """
    The Function projects flags of "source" onto flags of "field". Wherever the "field" flags are "better" then the
    source flags projected on them, they get overridden with this associated source flag value.
    
    Which "field"-flags are to be projected on which source flags, is controlled by the "method" and "freq"
    parameters.
    
    method: (field_flag in associated with "field", source_flags associated with "source")
    
    'inverse_nagg' - all field_flags within the range +/- freq/2 of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    'inverse_bagg' - all field_flags succeeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    'inverse_fagg' - all field_flags preceeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    
    'inverse_interpolation' - all field_flags within the range +/- freq of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    
    'inverse_nshift' - That field_flag within the range +/- freq/2, that is nearest to a source_flag, gets the source
        flags value. (if source_flag > field_flag)
    'inverse_bshift' - That field_flag succeeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    'inverse_nshift' - That field_flag preceeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    
    'match' - any field_flag with a timestamp matching a source_flags timestamp gets this source_flags value
    (if source_flag > field_flag)
    
    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to project the source-flags onto.
    
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift'}
        The method used for projection of source flags onto field flags. See description above for more details.
    
    source : str
        The source source of flags projection.
    
    freq : {None, str},default None
        The freq determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of source is used.
    """
    pass


def calculatePolynomialResidues(field, winsz, polydeg, numba, eval_flags, min_periods, flag):
    """
    Function fits a polynomial model to the data and returns the residues.
    
    The residue for value x is calculated by fitting a polynomial of degree "polydeg" to a data slice
    of size "winsz", wich has x at its center.
    
    Note, that the residues will be stored to the `field` field of the input data, so that the original data, the
    polynomial is fitted to, gets overridden.
    
    Note, that, if data[field] is not alligned to an equidistant frequency grid, the window size passed,
    has to be an offset string. Also numba boost options don`t apply for irregularly sampled
    timeseries.
    
    Note, that calculating the residues tends to be quite costy, because a function fitting is perfomed for every
    sample. To improve performance, consider the following possibillities:
    
    In case your data is sampled at an equidistant frequency grid:
    
    (1) If you know your data to have no significant number of missing values, or if you do not want to
        calculate residues for windows containing missing values any way, performance can be increased by setting
        min_periods=winsz.
    
    (2) If your data consists of more then around 200000 samples, setting numba=True, will boost the
        calculations up to a factor of 5 (for samplesize > 300000) - however for lower sample sizes,
        numba will slow down the calculations, also, up to a factor of 5, for sample_size < 50000.
        By default (numba='auto'), numba is set to true, if the data sample size exceeds 200000.
    
    in case your data is not sampled at an equidistant frequency grid:
    
    (1) Harmonization/resampling of your data will have a noticable impact on polyfittings performance - since
        numba_boost doesnt apply for irregularly sampled data in the current implementation.
    
    Note, that in the current implementation, the initial and final winsz/2 values do not get fitted.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
    winsz : {str, int}
        The size of the window you want to use for fitting. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension. The window will be centered around the vaule-to-be-fitted.
        For regularly sampled timeseries the period number will be casted down to an odd number if
        even.
    polydeg : int
        The degree of the polynomial used for fitting
    numba : {True, False, "auto"}, default "auto"
        Wheather or not to apply numbas just-in-time compilation onto the poly fit function. This will noticably
        increase the speed of calculation, if the sample size is sufficiently high.
        If "auto" is selected, numba compatible fit functions get applied for data consisiting of > 200000 samples.
    eval_flags : bool, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
    min_periods : {int, None}, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.
    flag : float, default BAD
        flag to set.
    """
    pass


def calculateRollingResidues():
    """
    TODO: docstring needed
    """
    pass


def roll(field, winsz, func, eval_flags, min_periods, center, flag):
    """
    Models the data with the rolling mean and returns the residues.
    
    Note, that the residues will be stored to the `field` field of the input data, so that the data that is modelled
    gets overridden.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
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
    flag : float, default BAD
        flag to set.
    """
    pass


def assignKNNScore(field, n_neighbors, trafo, trafo_on_partition, scoring_func, target_field, partition_freq, partition_min, kNN_algorithm, metric, p, radius):
    """
    TODO: docstring need a rework
    Score datapoints by an aggregation of the dictances to their k nearest neighbors.
    
    The function is a wrapper around the NearestNeighbors method from pythons sklearn library (See reference [1]).
    
    The steps taken to calculate the scores are as follows:
    
    1. All the timeseries, named fields, are combined to one feature space by an *inner* join on their date time indexes.
       thus, only samples, that share timestamps across all fields will be included in the feature space
    2. Any datapoint/sample, where one ore more of the features is invalid (=np.nan) will get excluded.
    3. For every data point, the distance to its `n_neighbors` nearest neighbors is calculated by applying the
       metric `metric` at grade `p` onto the feature space. The defaults lead to the euclidian to be applied.
       If `radius` is not None, it sets the upper bound of distance for a neighbor to be considered one of the
       `n_neigbors` nearest neighbors. Furthermore, the `partition_freq` argument determines wich samples can be
       included into a datapoints nearest neighbors list, by segmenting the data into chunks of specified temporal
       extension and feeding that chunks to the kNN algorithm seperatly.
    4. For every datapoint, the calculated nearest neighbors distances get aggregated to a score, by the function
       passed to `scoring_func`. The default, ``sum`` obviously just sums up the distances.
    5. The resulting timeseries of scores gets assigned to the field target_field.
    
    Parameters
    ----------
    field : str
        The reference variable, the deviation from wich determines the flagging.
    n_neighbors : int, default 10
        The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.
    trafo : Callable[np.array, np.array], default lambda x: x
        Transformation to apply on the variables before kNN scoring
    trafo_on_partition : bool, default True
        Weather or not to apply the transformation `trafo` onto the whole variable or onto each partition seperatly.
    scoring_func : Callable[numpy.array, float], default np.sum
        A function that assigns a score to every one dimensional array, containing the distances
        to every datapoints `n_neighbors` nearest neighbors.
    target_field : str, default 'kNN_scores'
        Name of the field, where the resulting scores should be written to.
    partition_freq : {np.inf, float, str}, default np.inf
        Determines the segmentation of the data into partitions, the kNN algorithm is
        applied onto individually.
    
        * ``np.inf``: Apply Scoring on whole data set at once
        * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
        * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
          string
    
    partition_min : int, default 2
        The minimum number of periods that have to be present in a partition for the kNN scoring
        to be applied. If the number of periods present is below `partition_min`, the score for the
        datapoints in that partition will be np.nan.
    kNN_algorithm : {'ball_tree', 'kd_tree', 'brute', 'auto'}, default 'ball_tree'
        The search algorithm to find each datapoints k nearest neighbors.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    metric : str, default 'minkowski'
        The metric the distances to any datapoints neighbors is computed with. The default of `metric`
        together with the default of `p` result in the euclidian to be applied.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    p : int, default 2
        The grade of the metrice specified by parameter `metric`.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    radius : {None, float}, default None
        If the radius is not None, only the distance to neighbors that ly within the range specified by `radius`
        are comprised in the scoring aggregation.
        The scoring method passed must be capable of handling np.nan values - since, for every point missing
        within `radius` range to make complete the list of the distances to the `n_neighbors` nearest neighbors,
        one np.nan value gets appended to the list passed to the scoring method.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    pass


def copy(field):
    """
    The function generates a copy of the data "field" and inserts it under the name field + suffix into the existing
    data.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to fork (copy).
    """
    pass


def drop(field):
    """
    The function drops field from the data dios and the flags.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to drop.
    """
    pass


def rename(field, new_name):
    """
    The function renames field to new name (in both, the flags and the data).
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to rename.
    new_name : str
        String, field is to be replaced with.
    """
    pass


def mask(field, mode, mask_var, period_start, period_end, include_bounds):
    """
    This function realizes masking within saqc.
    
    Due to some inner saqc mechanics, it is not straight forwardly possible to exclude
    values or datachunks from flagging routines. This function replaces flags with UNFLAGGED
    value, wherever values are to get masked. Furthermore, the masked values get replaced by
    np.nan, so that they dont effect calculations.
    
    Here comes a recipe on how to apply a flagging function only on a masked chunk of the variable field:
    
    1. dublicate "field" in the input data (copy)
    2. mask the dublicated data (mask)
    3. apply the tests you only want to be applied onto the masked data chunks (saqc_tests)
    4. project the flags, calculated on the dublicated and masked data onto the original field data
        (projectFlags or flagGeneric)
    5. drop the dublicated data (drop)
    
    To see an implemented example, checkout flagSeasonalRange in the saqc.functions module
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-masked.
    mode : {"periodic", "mask_var"}
        The masking mode.
        - "periodic": parameters "period_start", "period_end" are evaluated to generate a periodical mask
        - "mask_var": data[mask_var] is expected to be a boolean valued timeseries and is used as mask.
    mask_var : {None, str}, default None
        Only effective if mode == "mask_var"
        Fieldname of the column, holding the data that is to be used as mask. (must be moolean series)
        Neither the series` length nor its labels have to match data[field]`s index and length. An inner join of the
        indices will be calculated and values get masked where the values of the inner join are "True".
    period_start : {None, str}, default None
        Only effective if mode == "seasonal"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `period_end` parameter.
        See examples section below for some examples.
    period_end : {None, str}, default None
        Only effective if mode == "periodic"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `period_end` parameter.
        See examples section below for some examples.
    include_bounds : boolean
        Wheather or not to include the mask defining bounds to the mask.
    
    Examples
    --------
    The `period_start` and `period_end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
    They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
    (mm=month, dd=day, HH=hour, MM=minute, SS=second)
    Single digit specifications have to be given with leading zeros.
    `period_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
    The highest date unit gives the period.
    For example:
    
    >>> period_start = "01T15:00:00"
    >>> period_end = "13T17:30:00"
    
    Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked
    
    >>> period_start = "01:00"
    >>> period_end = "04:00"
    
    All the values between the first and 4th minute of every hour get masked.
    
    >>> period_start = "01-01T00:00:00"
    >>> period_end = "01-03T00:00:00"
    
    Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
    include 00:00:00 at the first of march. To exclude this one, pass:
    
    >>> period_start = "01-01T00:00:00"
    >>> period_end = "02-28T23:59:59"
    
    To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
    season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:
    
    >>> period_start = "22:00:00"
    >>> period_end = "06:00:00"
    
    When inclusive_selection="season", all above examples work the same way, only that you now
    determine wich values NOT TO mask (=wich values are to constitute the "seasons").
    """
    pass


def plot(field, save_path, max_gap, stats, plot_kwargs, fig_kwargs, save_kwargs):
    """
    Stores or shows a figure object, containing data graph with flag marks for field.
    
    Parameters
    ----------
    field : str
        Name of the variable-to-plot
    save_path : str, default ''
        Path to the location where the figure shall be stored to. If '' is passed, interactive mode is accessed instead
        of figure storage.
    max_gap : {None, str}, default None:
        If None, all the points in the data will be connected, resulting in long linear lines, where continous chunks
        of data is missing. (nans in the data get dropped before plotting.)
        If an Offset string is passed, only points that have a distance below `max_gap` get connected via the plotting
        line.
    stats : bool, default False
        Whether to include statistics table in plot.
    plot_kwargs : dict, default {}
        Keyword arguments controlling plot generation. Will be passed on to the ``Matplotlib.axes.Axes.set()`` property
        batch setter for the axes showing the data plot. The most relevant of those properties might be "ylabel",
        "title" and "ylim".
        In Addition, following options are available:
    
        * {'slice': s} property, that determines a chunk of the data to be plotted / processed. `s` can be anything,
          that is a valid argument to the ``pandas.Series.__getitem__`` method.
        * {'history': str}
            * str="all": All the flags are plotted with colored dots, refering to the tests they originate from
            * str="valid": - same as 'all' - but only plots those flags, that are not removed by later tests
    
    fig_kwargs : dict, default {"figsize": (16, 9)}
        Keyword arguments controlling figure generation.
    save_kwargs : dict, default {}
        Keywords to be passed on to the ``matplotlib.pyplot.savefig`` method, handling the figure storing.
        NOTE: To store an pickle, that can be used to regain an interactive figure window, use the option
        {'pickle': True}. This will result in all the other save_kwargs to be ignored.
        To enter interactive mode for a pickled figure, simply do: pickle.load(open(savepath,'w')).show()
    stats_dict: Optional[dict] = {}
        Dictionary of additional statisticts to write to the statistics table accompanying the data plot.
        (Only relevant if `stats`=True). An entry to the stats_dict has to be of the form:
    
        * {"stat_name": lambda x, y, z: func(x, y, z)}
    
        The lambda args ``x``,``y``,``z`` will be fed by:
    
        * ``x``: the data (``data[field]``).
        * ``y``: the flags (``flags[field]``).
        * ``z``: The passed flags level (``kwargs[flag]``)
    
        See examples section for examples
    
    Examples
    --------
    Summary statistic function examples:
    
    >>> func = lambda x, y, z: len(x)
    
    Total number of nan-values:
    
    >>> func = lambda x, y, z: x.isna().sum()
    
    Percentage of values, flagged greater than passed flag (always round float results to avoid table cell overflow):
    
    >>> func = lambda x, y, z: round((x.isna().sum()) / len(x), 2)
    """
    pass


def transform(field, func, partition_freq):
    """
    Function to transform data columns with a transformation that maps series onto series of the same length.
    
    Note, that flags get preserved.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-transformed.
    func : Callable[{pd.Series, np.array}, np.array]
        Function to transform data[field] with.
    partition_freq : {None, float, str}, default None
        Determines the segmentation of the data into partitions, the transformation is applied on individually
    
        * ``np.inf``: Apply transformation on whole data set at once
        * ``x`` > 0 : Apply transformation on successive data chunks of periods length ``x``
        * Offset String : Apply transformation on successive partitions of temporal extension matching the passed offset
          string
    """
    pass

