"""

"""
def interpolateByRolling(field, winsz, func, center, min_periods, interpol_flag):
    """
    Interpolates missing values (nan values present in the data) by assigning them the aggregation result of
    a window surrounding them.
    
    Note, that in the current implementation, center=True can only be used with integer window sizes - furthermore
    note, that integer window sizes can yield screwed aggregation results for not-harmonized or irregular data.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    winsz : int, str
        The size of the window, the aggregation is computed from. Either counted in periods number (Integer passed),
        or defined by a total temporal extension (offset String passed).
    func : Callable
        The function used for aggregation.
    center : bool, default True
        Wheather or not the window, the aggregation is computed of, is centered around the value to be interpolated.
    min_periods : int
        Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
        computed.
    interpol_flag : {'GOOD', 'BAD', 'UNFLAGGED', str}, default 'UNFLAGGED'
        Flag that is to be inserted for the interpolated values. You can either pass one of the three major flag-classes
        or specify directly a certain flag from the passed flagger.
    """
    pass


def interpolateInvalid(field, method, inter_order, inter_limit, interpol_flag, downgrade_interpolation, not_interpol_flags):
    """
    Function to interpolate nan values in the data.
    
    There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
    the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.
    
    Note, that the `inter_limit` keyword really restricts the interpolation to chunks, not containing more than
    `inter_limit` successive nan entries.
    
    Note, that the function differs from ``proc_interpolateGrid``, in its behaviour to ONLY interpolate nan values that
    were already present in the data passed.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    inter_order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    inter_limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated.
    interpol_flag : {'GOOD', 'BAD', 'UNFLAGGED', str}, default 'UNFLAGGED'
        Flag that is to be inserted for the interpolated values. You can either pass one of the three major flag-classes
        or specify directly a certain flag from the passed flagger.
    downgrade_interpolation : bool, default False
        If interpolation can not be performed at `inter_order` - (not enough values or not implemented at this order) -
        automaticalyy try to interpolate at order `inter_order` :math:`- 1`.
    not_interpol_flags : {None, str, List[str]}, default None
        A list of flags or a single Flag, marking values, you want NOT to be interpolated.
    """
    pass


def interpolateIndex(field, freq, method, inter_order, to_drop, downgrade_interpolation, empty_intervals_flag, grid_field, inter_limit, freq_check):
    """
    Function to interpolate the data at regular (equidistant) timestamps (or Grid points).
    
    Note, that the interpolation will only be calculated, for grid timestamps that have a preceding AND a succeeding
    valid data value within "freq" range.
    
    Note, that the function differs from proc_interpolateMissing, by returning a whole new data set, only containing
    samples at the interpolated, equidistant timestamps (of frequency "freq").
    
    Note, it is possible to interpolate unregular "grids" (with no frequencies). In fact, any date index
    can be target of the interpolation. Just pass the field name of the variable, holding the index
    you want to interpolate, to "grid_field". 'freq' is then use to determine the maximum gap size for
    a grid point to be interpolated.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-interpolated.
    freq : str
        An Offset String, interpreted as the frequency of
        the grid you want to interpolate your data at.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    inter_order : integer, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.
    to_drop : {None, str, List[str]}, default None
        Flags that refer to values you want to drop before interpolation - effectively excluding grid points from
        interpolation, that are only surrounded by values having a flag in them, that is listed in drop flags. Default
        results in the flaggers *BAD* flag to be the drop_flag.
    downgrade_interpolation : bool, default False
        If interpolation can not be performed at `inter_order` - (not enough values or not implemented at this order) -
        automatically try to interpolate at order `inter_order` :math:`- 1`.
    empty_intervals_flag : str, default None
        A Flag, that you want to assign to those values in the resulting equidistant sample grid, that were not
        surrounded by valid data in the original dataset, and thus were not interpolated. Default automatically assigns
        ``flagger.BAD`` flag to those values.
    grid_field : String, default None
        Use the timestamp of another variable as (not necessarily regular) "grid" to be interpolated.
    inter_limit : Integer, default 2
        Maximum number of consecutive Grid values allowed for interpolation. If set
        to *n*, chunks of *n* and more consecutive grid values, where there is no value in between, wont be
        interpolated.
    freq_check : {None, 'check', 'auto'}, default None
    
        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
          if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)
    """
    pass

