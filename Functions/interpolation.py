"""

"""


def interpolateByRolling(field, window, func, center, min_periods, target, flag):
    """
    Interpolates nan-values in the data by assigning them the aggregation result of the window surrounding them.

    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-interpolated.

    window : int, str
        The size of the window, the aggregation is computed from. An integer define the number of periods to be used,
        an string is interpreted as an offset. ( see `pandas.rolling` for more information).
        Integer windows may result in screwed aggregations if called on none-harmonized or irregular data.

    func : Callable
        The function used for aggregation.

    center : bool, default True
        Center the window around the value. Can only be used with integer windows, otherwise it is silently ignored.

    min_periods : int
        Minimum number of valid (not np.nan) values that have to be available in a window for its aggregation to be
        computed.

    target : str
        Write the calculation results to ``target``. (keeping flags).

    flag : float or None, default UNFLAGGED
        Flag that is to be inserted for the interpolated values. If ``None`` no flags are set.
    """
    pass


def interpolateInvalid(field, method, order, limit, flag, downgrade):
    """
    Function to interpolate nan values in the data.

    There are available all the interpolation methods from the pandas.interpolate method and they are applicable by
    the very same key words, that you would pass to the ``pd.Series.interpolate``'s method parameter.

    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-interpolated.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
        The interpolation method to use.

    order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `limit` successive nan entries.

    flag : float or None, default UNFLAGGED
        Flag that is set for interpolated values. If ``None``, no flags are set at all.

    downgrade : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``order``, or
        simply because not enough values are present in a interval.
    """
    pass


def interpolateIndex(field, freq, method, order, limit, downgrade, target):
    """
    Function to interpolate the data at regular (equidistant) timestamps (or Grid points).

    Note, that the interpolation will only be calculated, for grid timestamps that have a preceding AND a succeeding
    valid data value within "freq" range.

    Parameters
    ----------
    field : str
        Name of the column, holding the data-to-be-interpolated.

    freq : str
        An Offset String, interpreted as the frequency of
        the grid you want to interpolate your data at.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.

    order : int, default 2
        If there your selected interpolation method can be performed at different 'orders' - here you pass the desired
        order.

    limit : int, default 2
        Maximum number of consecutive 'nan' values allowed for a gap to be interpolated. This really restricts the
        interpolation to chunks, containing not more than `limit` successive nan entries.

    downgrade : bool, default False
        If `True` and the interpolation can not be performed at current order, retry with a lower order.
        This can happen, because the chosen ``method`` does not support the passed ``order``, or
        simply because not enough values are present in a interval.

    target : str, default None
        Write result to ``target``.

    """
    pass
