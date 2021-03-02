"""

"""
def fitPolynomial(field, winsz, polydeg, numba, eval_flags, min_periods, return_residues):
    """
    Function fits a polynomial model to the data and returns the fitted data curve.
    
    The fit is calculated by fitting a polynomial of degree `polydeg` to a data slice
    of size `winsz`, that has x at its center.
    
    Note, that the resulting fit is stored to the `field` field of the input data, so that the original data, the
    polynomial is fitted to, gets overridden.
    
    Note, that, if data[field] is not alligned to an equidistant frequency grid, the window size passed,
    has to be an offset string. Also numba boost options don`t apply for irregularly sampled
    timeseries.
    
    Note, that calculating the residues tends to be quite costy, because a function fitting is perfomed for every
    sample. To improve performance, consider the following possibillities:
    
    In case your data is sampled at an equidistant frequency grid:
    
    (1) If you know your data to have no significant number of missing values, or if you do not want to
        calculate residues for windows containing missing values any way, performance can be increased by setting
        min_periods=winsz.
    
    (2) If your data consists of more then around 200000 samples, setting numba=True, will boost the
        calculations up to a factor of 5 (for samplesize > 300000) - however for lower sample sizes,
        numba will slow down the calculations, also, up to a factor of 5, for sample_size < 50000.
        By default (numba='auto'), numba is set to true, if the data sample size exceeds 200000.
    
    in case your data is not sampled at an equidistant frequency grid:
    
    (1) Harmonization/resampling of your data will have a noticable impact on polyfittings performance - since
        numba_boost doesnt apply for irregularly sampled data in the current implementation.
    
    Note, that in the current implementation, the initial and final winsz/2 values do not get fitted.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-modelled.
    winsz : {str, int}
        The size of the window you want to use for fitting. If an integer is passed, the size
        refers to the number of periods for every fitting window. If an offset string is passed,
        the size refers to the total temporal extension. The window will be centered around the vaule-to-be-fitted.
        For regularly sampled timeseries the period number will be casted down to an odd number if
        even.
    polydeg : int
        The degree of the polynomial used for fitting
    numba : {True, False, "auto"}, default "auto"
        Wheather or not to apply numbas just-in-time compilation onto the poly fit function. This will noticably
        increase the speed of calculation, if the sample size is sufficiently high.
        If "auto" is selected, numba compatible fit functions get applied for data consisiting of > 200000 samples.
    eval_flags : bool, default True
        Wheather or not to assign new flags to the calculated residuals. If True, a residual gets assigned the worst
        flag present in the interval, the data for its calculation was obtained from.
    min_periods : {int, None}, default 0
        The minimum number of periods, that has to be available in every values fitting surrounding for the polynomial
        fit to be performed. If there are not enough values, np.nan gets assigned. Default (0) results in fitting
        regardless of the number of values present (results in overfitting for too sparse intervals). To automatically
        set the minimum number of periods to the number of values in an offset defined window size, pass np.nan.
    return_residues : bool, default False
        Internal parameter. Makes the method return the residues instead of the fit.
    """
    pass

