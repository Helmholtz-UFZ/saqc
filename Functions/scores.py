"""

"""


def assignKNNScore(field, n, func, freq, min_periods, method, metric, p, radius):
    """
    TODO: docstring need a rework
    Score datapoints by an aggregation of the dictances to their k nearest neighbors.

    The function is a wrapper around the NearestNeighbors method from pythons sklearn library (See reference [1]).

    The steps taken to calculate the scores are as follows:

    1. All the timeseries, given through ``field``, are combined to one feature space by an *inner* join on their
       date time indexes. thus, only samples, that share timestamps across all ``field``s will be included in the
       feature space.
    2. Any datapoint/sample, where one ore more of the features is invalid (=np.nan) will get excluded.
    3. For every data point, the distance to its `n` nearest neighbors is calculated by applying the
       metric `metric` at grade `p` onto the feature space. The defaults lead to the euclidian to be applied.
       If `radius` is not None, it sets the upper bound of distance for a neighbor to be considered one of the
       `n` nearest neighbors. Furthermore, the `freq` argument determines wich samples can be
       included into a datapoints nearest neighbors list, by segmenting the data into chunks of specified temporal
       extension and feeding that chunks to the kNN algorithm seperatly.
    4. For every datapoint, the calculated nearest neighbors distances get aggregated to a score, by the function
       passed to `func`. The default, ``sum`` obviously just sums up the distances.
    5. The resulting timeseries of scores gets assigned to the field target.

    Parameters
    ----------
    field : list of str
        input variable names.
    n : int, default 10
        The number of nearest neighbors to which the distance is comprised in every datapoints scoring calculation.
    func : Callable[numpy.array, float], default np.sum
        A function that assigns a score to every one dimensional array, containing the distances
        to every datapoints `n` nearest neighbors.
    freq : {np.inf, float, str}, default np.inf
        Determines the segmentation of the data into partitions, the kNN algorithm is
        applied onto individually.

        * ``np.inf``: Apply Scoring on whole data set at once
        * ``x`` > 0 : Apply scoring on successive data chunks of periods length ``x``
        * Offset String : Apply scoring on successive partitions of temporal extension matching the passed offset
          string

    min_periods : int, default 2
        The minimum number of periods that have to be present in a partition for the kNN scoring
        to be applied. If the number of periods present is below `min_periods`, the score for the
        datapoints in that partition will be np.nan.
    method : {'ball_tree', 'kd_tree', 'brute', 'auto'}, default 'ball_tree'
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
        within `radius` range to make complete the list of the distances to the `n` nearest neighbors,
        one np.nan value gets appended to the list passed to the scoring method.
        The keyword just gets passed on to the underlying sklearn method.
        See reference [1] for more information on the algorithm.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    pass
