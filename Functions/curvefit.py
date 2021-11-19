"""

"""
def fitPolynomial(field, window, order, set_flags, min_periods, return_residues, target, flag):
    """
    Function fits a polynomial model to the data and returns the fitted data curve.
    
    The fit is calculated by fitting a polynomial of degree `order` to a data slice
    of size `window`, that has x at its center.
    
    Note, that the resulting fit is stored to the `field` field of the input data, so that the original data, the
    polynomial is fitted to, gets overridden.
    
    Note, that, if data[field] is not alligned to an equidistant frequency grid, the window size passed,
    has to be an offset string.
    
    Note, that calculating the residues tends to be quite costy, because a function fitting is perfomed for every
    sample. To improve performance, consider the following possibillities:
    
    In case your data is sampled at an equidistant frequency grid:
    
    (1) If you know your data to have no significant number of missing values, or if you do not want to
        calculate residues for windows containing missing values any way, performance can be increased by setting
        min_periods=window.
    
    Note, that in the current implementation, the initial and final window/2 values do not get fitted.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
    window : {str, int}
        The size of the window you want to use for fitting. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension. The window will be centered around the vaule-to-be-fitted.
        For regularly sampled timeseries the period number will be casted down to an odd number if
        even.
    order : int
        The degree of the polynomial used for fitting
    set_flags : bool, default True
        Whether or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
    min_periods : {int, None}, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.
    return_residues : bool, default False
        Internal parameter. Makes the method return the residues instead of the fit.
    target : str or None, default None
        Write the reult of the processing to another variable then, ``field``. Must not already exist.
    flag : float, default BAD
        flag to set.
    """
    pass

