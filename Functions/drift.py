"""

"""
def flagDriftFromNorm(field, freq, spread, frac, metric, method, flag):
    """
    The function flags value courses that significantly deviate from a group of normal value courses.
    
    "Normality" is determined in terms of a maximum spreading distance, that members of a normal group must not exceed.
    In addition, only a group is considered "normal" if it contains more then `frac` percent of the
    variables in "field".
    
    See the Notes section for a more detailed presentation of the algorithm
    
    Parameters
    ----------
    field : list of str
        List of fieldnames in data, determining which variables are to be included into the flagging process.
    freq : str
        An offset string, determining the size of the seperate datachunks that the algorihm is to be piecewise
        applied on.
    spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the "normal" group. See Notes section
        for more details.
    frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables, the "normal" group has to comprise to be the
        normal group actually. The higher that value, the more stable the algorithm will be with respect to false
        positives. Also, nobody knows what happens, if this value is below 0.5.
    metric : Callable[[numpy.array, numpy.array], float]
        A distance function. It should be a function of 2 1-dimensional arrays and return a float scalar value.
        This value is interpreted as the distance of the two input arrays. The default is the averaged manhatten metric.
        See the Notes section to get an idea of why this could be a good choice.
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the timeseries.
        See the Notes section for more details.
        The keyword gets passed on to scipy.hierarchy.linkage. See its documentation to learn more about the different
        keywords (References [1]).
        See wikipedia for an introduction to hierarchical clustering (References [2]).
    flag : float, default BAD
        flag to set.
    
    Notes
    -----
    following steps are performed for every data "segment" of length `freq` in order to find the
    "abnormal" data:
    
    1. Calculate the distances :math:`d(x_i,x_j)` for all :math:`x_i` in parameter `field`. (with :math:`d`
       denoting the distance function
       passed to the parameter `metric`.
    2. Calculate a dendogram with a hierarchical linkage algorithm, specified by the parameter `method`.
    3. Flatten the dendogram at the level, the agglomeration costs exceed the value given by the parameter `spread`
    4. check if there is a cluster containing more than `frac` percentage of the variables in field.
    
        1. if yes: flag all the variables that are not in that cluster (inside the segment)
        2. if no: flag nothing
    
    The main parameter giving control over the algorithms behavior is the `spread` parameter, that determines
    the maximum spread of a normal group by limiting the costs, a cluster agglomeration must not exceed in every
    linkage step.
    For singleton clusters, that costs just equal half the distance, the timeseries in the clusters, have to
    each other. So, no timeseries can be clustered together, that are more then
    2*`spread` distanted from each other.
    When timeseries get clustered together, this new clusters distance to all the other timeseries/clusters is
    calculated according to the linkage method specified by `method`. By default, it is the minimum distance,
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


def flagDriftFromReference(field, reference, freq, thresh, metric, flag):
    """
    The function flags value courses that deviate from a reference course by a margin exceeding a certain threshold.
    
    The deviation is measured by the distance function passed to parameter metric.
    
    Parameters
    ----------
    field : list of str
        List of fieldnames in data, determining wich variables are to be included into the flagging process.
    reference : str
        The reference variable, the deviation from wich determines the flagging.
    freq : str
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


def flagDriftFromScaledNorm(field, set_1, set_2, freq, spread, frac, metric, method, flag):
    """
    The function linearly rescales one set of variables to another set of variables
    with a different scale and then flags value courses that significantly deviate
    from a group of normal value courses.
    
    The two sets of variables can be linearly scaled one to another and hence the
    scaling transformation is performed via linear regression: A linear regression is
    performed on each pair of variables giving a slope and an intercept. The
    transformation is then calculated a the median of all the calculated slopes and
    intercepts.
    
    Once the transformation is performed, the function flags those values,
    that deviate from a group of normal values. "Normality" is determined in terms of
    a maximum spreading distance, that members of a normal group must not exceed. In
    addition, only a group is considered "normal" if it contains more then `frac`
    percent of the variables in "fields".
    
    Parameters
    ----------
    field : str
        A dummy parameter.
    
    set_1 : str
        The first set of fieldnames in data to be included into the flagging process.
    
    set_2 : str
        The second set of fieldnames in data to be included into the flagging process.
    
    freq : str
        An offset string, determining the size of the seperate datachunks that the
        algorihm is to be piecewise applied on.
    
    spread : float
        A parameter limiting the maximum "spread" of the timeseries, allowed in the
        "normal" group. See Notes section for more details.
    
    frac : float, default 0.5
        Has to be in [0,1]. Determines the minimum percentage of variables,
        the "normal" group has to comprise to be the normal group actually. The
        higher that value, the more stable the algorithm will be with respect to
        false positives. Also, nobody knows what happens, if this value is below 0.5.
    
    metric : Callable[(numpyp.array, numpy-array), float]
        A distance function. It should be a function of 2 1-dimensional arrays and
        return a float scalar value. This value is interpreted as the distance of the
        two input arrays. The default is the averaged manhatten metric. See the Notes
        section to get an idea of why this could be a good choice.
    
    method : str, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the
        timeseries. See the Notes section for more details. The keyword gets passed
        on to scipy.hierarchy.linkage. See its documentation to learn more about the
        different keywords (References [1]). See wikipedia for an introduction to
        hierarchical clustering (References [2]).
    
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


def correctDrift(field, maintenance_field, model, cal_range, target, flag):
    """
    The function corrects drifting behavior.
    
    See the Notes section for an overview over the correction algorithm.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to correct.
    maintenance_field : str
        The fieldname of the datacolumn holding the support-points information.
        The maint data is to expected to have following form:
        The series' timestamp itself represents the beginning of a
        maintenance event, wheras the values represent the endings of the maintenance intervals.
    model : Callable or {'exponential', 'linear'}
        A modelfunction describing the drift behavior, that is to be corrected.
        Either use built-in exponential or linear drift model by passing a string, or pass a custom callable.
        The model function must always contain the keyword parameters 'origin' and 'target'.
        The starting parameter must always be the parameter, by wich the data is passed to the model.
        After the data parameter, there can occure an arbitrary number of model calibration arguments in
        the signature.
        See the Notes section for an extensive description.
    cal_range : int, default 5
        The number of values the mean is computed over, for obtaining the value level directly after and
        directly before maintenance event. This values are needed for shift calibration. (see above description)
    target : str or None, default None
        Write the reult of the processing to another variable then, ``field``. Must not already exist.
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


def correctRegimeAnomaly(field, cluster_field, model, tolerance, epoch, target):
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
    tolerance : {None, str}, default None:
        If an offset string is passed, a data chunk of length `offset` right at the
        start and right at the end is ignored when fitting the model. This is to account for the
        unreliability of data near the changepoints of regimes.
    epoch : bool, default False
        If True, use "seconds from epoch" as x input to the model func, instead of "seconds from regime start".
    target : str or None, default None
        Write the reult of the processing to another variable then, ``field``. Must not already exist.
    
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
    max_jump : float
        when searching for changepoints in mean - this is the threshold a mean difference in the
        sliding window search must exceed to trigger changepoint detection.
    spread : float
        threshold denoting the maximum, regimes are allowed to abolutely differ in their means
        to form the "normal group" of values.
    window : str
        Size of the adjacent windows that are used to search for the mean changepoints.
    min_periods : int
        Minimum number of periods a search window has to contain, for the result of the changepoint
        detection to be considered valid.
    tolerance : {None, str}, default None:
        If an offset string is passed, a data chunk of length `offset` right from the
        start and right before the end of any regime is ignored when calculating a regimes mean for data correcture.
        This is to account for the unrelyability of data near the changepoints of regimes.
    target : str or None, default None
        Write the result of the processing to another variable then, ``field``. Must not already exist.
    
    """
    pass


def flagRegimeAnomaly(field, cluster_field, spread, method, metric, frac, flag):
    """
    A function to flag values belonging to an anomalous regime regarding modelling regimes of field.
    
    "Normality" is determined in terms of a maximum spreading distance, regimes must not exceed in respect
    to a certain metric and linkage method.
    
    In addition, only a range of regimes is considered "normal", if it models more then `frac` percentage of
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
    spread : float
        A threshold denoting the valuelevel, up to wich clusters a agglomerated.
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the variables.
    metric : Callable[[numpy.array, numpy.array], float], default lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes. Defaults to just the difference in mean.
    frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.
    flag : float, default BAD
        flag to set.
    """
    pass


def assignRegimeAnomaly(field, cluster_field, spread, method, metric, frac, set_cluster, set_flags, flag):
    """
    A function to detect values belonging to an anomalous regime regarding modelling
    regimes of field.
    
    The function changes the value of the regime cluster labels to be negative.
    "Normality" is determined in terms of a maximum spreading distance, regimes must
    not exceed in respect to a certain metric and linkage method. In addition,
    only a range of regimes is considered "normal", if it models more then `frac`
    percentage of the valid samples in "field". Note, that you must detect the regime
    changepoints prior to calling this function. (They are expected to be stored
    parameter `cluster_field`.)
    
    Note, that it is possible to perform hypothesis tests for regime equality by
    passing the metric a function for p-value calculation and selecting linkage
    method "complete".
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    cluster_field : str
        The name of the column in data, holding the cluster labels for the samples in field. (has to be indexed
        equal to field)
    spread : float
        A threshold denoting the valuelevel, up to wich clusters a agglomerated.
    method : str, default "single"
        The linkage method used for hierarchical (agglomerative) clustering of the
        variables.
    metric : Callable[[numpy.array, numpy.array], float], default lambda x, y: np.abs(np.nanmean(x) - np.nanmean(y))
        A metric function for calculating the dissimilarity between 2 regimes. Defaults
        to just the difference in mean.
    frac : float
        Has to be in [0,1]. Determines the minimum percentage of samples,
        the "normal" group has to comprise to be the normal group actually.
    set_cluster : bool, default False
        If True, all data, considered "anormal", gets assigned a negative clusterlabel.
         This option is present for further use (correction) of the anomaly information.
    set_flags : bool, default True
        Whether or not to flag abnormal values (do not flag them, if you want to
        correct them afterwards, because flagged values usually are not visible in
        further tests).
    flag : float, default BAD
        flag to set.
    """
    pass

