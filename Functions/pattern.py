"""

"""


def flagPatternByWavelet(field):
    """
    Pattern recognition via wavelets.

    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the wavelet algorithm as
        presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged

    Parameters
    ----------

    field : str
        The fieldname of the data column, you want to correct.
    """
    pass


def calculateDistanceByDTW(reference, normalize):
    """
    Calculate the DTW-distance of data to pattern in a rolling calculation.

    The data is compared to pattern in a rolling window.
    The size of the rolling window is determined by the timespan defined
    by the first and last timestamp of the reference data's datetime index.

    For details see the linked functions in the `See Also` section.

    Parameters
    ----------
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

    Notes
    -----
    The data must be regularly sampled, otherwise a ValueError is raised.
    NaNs in the data will be dropped before dtw distance calculation.

    See Also
    --------
    flagPatternByDTW : flag data by DTW
    """
    pass


def flagPatternByDTW(field, reference, max_distance, normalize):
    """
    Pattern Recognition via Dynamic Time Warping.

    The steps are:
     1. work on a moving window
     2. for each data chunk extracted from each window, a distance to the given pattern
        is calculated, by the dynamic time warping algorithm [1]
     3. if the distance is below the threshold, all the data in the window gets flagged

    Parameters
    ----------
    field : str
        The name of the data column

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

    Notes
    -----
    The window size of the moving window is set to equal the temporal extension of the
    reference datas datetime index.

    References
    ----------
    Find a nice description of underlying the Dynamic Time Warping Algorithm here:

    [1] https://cran.r-project.org/web/packages/dtw/dtw.pdf
    """
    pass
