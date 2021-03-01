"""

"""
def flagByStray(field, partition_freq, partition_min, iter_start, alpha):
    """
    Flag outliers in 1-dimensional (score) data with the STRAY Algorithm.
    
    Find more information on the algorithm in References [1].
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    partition_freq : {None, str, int}, default None
        partition_freq : {np.inf, float, str}, default np.inf
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
    pass


def flagMVScores(field, fields, trafo, alpha, n_neighbors, scoring_func, iter_start, stray_partition, stray_partition_min, trafo_on_partition, reduction_range, reduction_drop_flagged, reduction_thresh, reduction_min_periods):
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


def flagRaise(field, thresh, raise_window, intended_freq, average_window, mean_raise_factor, min_slope, min_slope_weight, numba_boost):
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


def flagMAD(field, window):
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
    
    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    pass


def flagOffset(field, thresh, tolerance, window, numba_kickin):
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
    numba_kickin : int, default 200000
        When there are detected more than `numba_kickin` incidents of potential spikes,
        the pandas.rolling - part of computation gets "jitted" with numba.
        Default value hast proven to be around the break even point between "jit-boost" and "jit-costs".
    
    
    References
    ----------
    The implementation is a time-window based version of an outlier test from the UFZ Python library,
    that can be found here:
    
    https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py
    """
    pass


def flagByGrubbs(field, winsz, alpha, min_periods):
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
    
    References
    ----------
    introduction to the grubbs test:
    
    [1] https://en.wikipedia.org/wiki/Grubbs%27s_test_for_outliers
    """
    pass


def flagRange(field, min, max):
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
    """
    pass


def flagCrossStatistic(field, fields, thresh, cross_stat):
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
    
    References
    ----------
    [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    pass

