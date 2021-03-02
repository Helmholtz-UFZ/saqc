"""

"""
def flagPatternByDTW(field):
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

