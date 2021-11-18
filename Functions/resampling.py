"""

"""


def linear(field, freq, target):
    """
    A method to "regularize" data by interpolating linearly the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``UNFLAGGED``.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.

    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.

    target : str, default None
        Write result to ``target``.
    """
    pass


def interpolate(field, freq, method, order, target):
    """
    A method to "regularize" data by interpolating the data at regular timestamp.

    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).

    Interpolated values will get assigned the worst flag within freq-range.

    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.

    Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
    and NOT ``'linear'``.

    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``UNFLAGGED``.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.

    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.

    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}
        The interpolation method you want to apply.

    order : int, default 1
        If your selected interpolation method can be performed at different *orders* - here you pass the desired
        order.

    target : str, default None
        Write result to ``target``.
    """
    pass


def shift(field, freq, method, freq_check, target):
    """
    Function to shift data and flags to a regular (equidistant) timestamp grid, according to ``method``.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-shifted.

    freq : str
        An frequency Offset String that will be interpreted as the sampling rate you want the data to be shifted to.

    method : {'fshift', 'bshift', 'nshift'}, default 'nshift'
        Specifies how misaligned data-points get propagated to a grid timestamp.
        Following choices are available:

        * 'nshift' : every grid point gets assigned the nearest value in its range. (range = +/- 0.5 * `freq`)
        * 'bshift' : every grid point gets assigned its first succeeding value, if one is available in
          the succeeding sampling interval.
        * 'fshift' : every grid point gets assigned its ultimately preceding value, if one is available in
          the preceeding sampling interval.

    freq_check : {None, 'check', 'auto'}, default None

        * ``None`` : do not validate frequency-string passed to `freq`
        * 'check' : estimate frequency and log a warning if estimate miss matches frequency string passed to `freq`,
          or if no uniform sampling rate could be estimated
        * 'auto' : estimate frequency and use estimate. (Ignores `freq` parameter.)

    target : str, default None
        Write result to ``target``.
    """
    pass


def resample(
    field,
    freq,
    func,
    maxna,
    maxna_group,
    maxna_flags,
    maxna_group_flags,
    flag_func,
    freq_check,
    target,
):
    """
    Function to resample the data.

    The data will be sampled at regular (equidistant) timestamps aka. Grid points.
    Sampling intervals therefore get aggregated with a function, specified by
    'agg_func' parameter and the result gets projected onto the new timestamps with a
    method, specified by "method". The following method (keywords) are available:

    * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get
        aggregated with agg_func and assigned to it.
    * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and
        the result gets assigned to the last grid point.
    * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and
        the result gets assigned to the next grid point.


    Note, that. if possible, functions passed to agg_func will get projected
    internally onto pandas.resample methods, wich results in some reasonable
    performance boost - however, for this to work, you should pass functions that
    have the __name__ attribute initialised and the according methods name assigned
    to it. Furthermore, you shouldnt pass numpys nan-functions (``nansum``,
    ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and
    they will thus not trigger ``resample.func()``, but the slower ``resample.apply(
    nanfunc)``. Also, internally, no nans get passed to the functions anyway,
    so that there is no point in passing the nan functions.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-resampled.

    freq : str
        An Offset String, that will be interpreted as the frequency you want to
        resample your data with.

    func : Callable
        The function you want to use for aggregation.

    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceding,
        succeeding or "surrounding" interval). See description above for more details.

    maxna : {None, int}, default None
        Maximum number NaNs in a resampling interval. If maxna is exceeded, the interval
        is set entirely to NaN.

    maxna_group : {None, int}, default None
        Same as `maxna` but for consecutive NaNs.

    maxna_flags : {None, int}, default None
        Same as `max_invalid`, only applying for the flags. The flag regarded
        as "invalid" value, is the one passed to empty_intervals_flag (
        default=``BAD``). Also this is the flag assigned to invalid/empty intervals.

    maxna_group_flags : {None, int}, default None
        Same as `maxna_flags`, only applying onto flags. The flag regarded as
        "invalid" value, is the one passed to empty_intervals_flag. Also this is the
        flag assigned to invalid/empty intervals.

    flag_func : Callable, default: max
        The function you want to aggregate the flags with. It should be capable of
        operating on the flags dtype (usually ordered categorical).

    freq_check : {None, 'check', 'auto'}, default None

        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs
            frequency string passed to 'freq', or if no uniform sampling rate could be
            estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)

    target : str, default None
        Write result to ``target``.
    """
    pass


def concatFlags(field, target, method, freq, drop):
    """
    The Function appends flags history of ``fields`` to flags history of ``target``.
    Before Appending, columns in ``field`` history are projected onto the target index via ``method``

    method: (field_flag in associated with "field", source_flags associated with "source")

    'inverse_nagg' - all target_flags within the range +/- freq/2 of a field_flag, get assigned this field flags value.
        (if field_flag > target_flag)
    'inverse_bagg' - all target_flags succeeding a field_flag within the range of "freq", get assigned this field flags
        value. (if field_flag > target_flag)
    'inverse_fagg' - all target_flags preceeding a field_flag within the range of "freq", get assigned this field flags
        value. (if field_flag > target_flag)

    'inverse_interpolation' - all target_flags within the range +/- freq of a field_flag, get assigned this source flags value.
        (if field_flag > target_flag)

    'inverse_nshift' - That target_flag within the range +/- freq/2, that is nearest to a field_flag, gets the source
        flags value. (if field_flag > target_flag)
    'inverse_bshift' - That target_flag succeeding a field flag within the range freq, that is nearest to a
        field_flag, gets assigned this field flags value. (if field_flag > target_flag)
    'inverse_nshift' - That target_flag preceeding a field flag within the range freq, that is nearest to a
        field_flag, gets assigned this field flags value. (if field_flag > target_flag)

    'match' - any target_flag with a timestamp matching a field_flags timestamp gets this field_flags value
    (if field_flag > target_flag)

    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.

    Parameters
    ----------
    field : str
        Fieldname of flags history to append.

    target : str
        Fieldname of flags history to append to.

    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
             'match'}
        The method used for projection of ``field`` flags onto ``target`` flags. See description above for more details.

    freq : {None, str},default None
        The ``freq`` determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of ``field`` is used.

    drop : default False
        If set to `True`, the `field` column will be removed after processing
    """
    pass
