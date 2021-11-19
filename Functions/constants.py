"""

"""


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


def flagByVariance(field, window, thresh, maxna, maxna_group, flag):
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
    maxna : int, default None
        Maximum number of NaNs tolerated in an interval. If more NaNs are present, the
        interval is not flagged as plateau.
    maxna_group : int, default None
        Same as `maxna` but for consecutive NaNs.
    flag : float, default BAD
        flag to set.
    """
    pass
