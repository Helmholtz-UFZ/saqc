"""
Detecting breakish changes in timeseries value courses.

This module provides functions to detect and flag  breakish changes in the data value course, like gaps
(:py:func:`flagMissing`), jumps/drops (:py:func:`flagJumps`) or isolated values (:py:func:`flagIsolated`).
"""
def flagMissing(field, nodata):
    """
    The function flags all values indicating missing data.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    nodata : any, default np.nan
        A value that defines missing data.
    """
    pass


def flagIsolated(field, gap_window, group_window):
    """
    The function flags arbitrary large groups of values, if they are surrounded by sufficiently
    large data gaps.
    
    A gap is a timespan containing either no data or invalid (usually `nan`) and flagged data only.
    
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
    
    Notes
    -----
    A series of values :math:`x_k,x_{k+1},...,x_{k+n}`, with associated timestamps :math:`t_k,t_{k+1},...,t_{k+n}`,
    is considered to be isolated, if:
    
    1. :math:`t_{k+1} - t_n <` `group_window`
    2. None of the :math:`x_j` with :math:`0 < t_k - t_j <` `gap_window`, is valid or unflagged (preceeding gap).
    3. None of the :math:`x_j` with :math:`0 < t_j - t_(k+n) <` `gap_window`, is valid or unflagged (succeding gap).
    
    See Also
    --------
    :py:func:`flagMissing`
    """
    pass


def flagJumps(field, thresh, winsz, min_periods):
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
    """
    pass

