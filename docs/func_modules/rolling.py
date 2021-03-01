"""

"""
def roll(field, winsz, func, eval_flags, min_periods, center):
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
    """
    pass

