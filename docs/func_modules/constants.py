"""

"""
def flagConstants(field, thresh, window):
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
        The fieldname of the column, holding the data-to-be-flagged.
    thresh : float
        Upper bound for the maximum total change of an interval to be flagged constant.
    window : str
        Lower bound for the size of an interval to be flagged constant.
    """
    pass


def flagByVariance(field, window, thresh, max_missing, max_consec_missing):
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
    """
    pass

