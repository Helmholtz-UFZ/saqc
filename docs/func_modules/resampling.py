"""

"""
def aggregate(field, freq, value_func, flag_func, method, to_drop):
    """
    A method to "regularize" data by aggregating (resampling) data at a regular timestamp.
    
    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).
    
    The data will therefor get aggregated with a function, specified by the `value_func` parameter and
    the result gets projected onto the new timestamps with a method, specified by "method".
    
    The following method (keywords) are available:
    
    * ``'nagg'``: (aggreagtion to nearest) - all values in the range (+/- freq/2) of a grid point get aggregated with
      `agg_func`. and assigned to it. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'bagg'``: (backwards aggregation) - all values in a sampling interval get aggregated with agg_func and the
      result gets assigned to the last regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.
    * ``'fagg'``: (forward aggregation) - all values in a sampling interval get aggregated with agg_func and the result
      gets assigned to the next regular timestamp. Flags get aggregated by `flag_func` and assigned the same way.
    
    Note, that, if there is no valid data (exisitng and not-na) available in a sampling interval assigned to a regular
    timestamp by the selected method, nan gets assigned to this timestamp. The associated flag will be of value
    ``flagger.UNFLAGGED``.
    
    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    freq : str
        The sampling frequency the data is to be aggregated (resampled) at.
    value_func : Callable
        The function you want to use for aggregation.
    flag_func : Callable
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).
    method : {'fagg', 'bagg', 'nagg'}, default 'nagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceeding, succeeding or
        "surrounding" interval). See description above for more details.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before aggregation - effectively excluding values that are flagged
        with a flag in to_drop from the aggregation process. Default results in flagger.BAD
        values being dropped initially.
    """
    pass


def linear(field, freq, to_drop):
    """
    A method to "regularize" data by interpolating linearly the data at regular timestamp.
    
    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).
    
    Interpolated values will get assigned the worst flag within freq-range.
    
    Note: the method will likely and significantly alter values and shape of ``data[field]``. The original data is kept
    in the data dios and assigned to the fieldname ``field + '_original'``.
    
    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``flagger.UNFLAGGED``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in to_drop from the interpolation process. Default results in flagger.BAD
        values being dropped initially.
    """
    pass


def interpolate(field, freq, method, order, to_drop):
    """
    A method to "regularize" data by interpolating the data at regular timestamp.
    
    A series of data is considered "regular", if it is sampled regularly (= having uniform sampling rate).
    
    Interpolated values will get assigned the worst flag within freq-range.
    
    There are available all the interpolations from the pandas.Series.interpolate method and they are called by
    the very same keywords.
    
    Note, that, to perform a timestamp aware, linear interpolation, you have to pass ``'time'`` as `method`,
    and NOT ``'linear'``.
    
    Note: the `method` will likely and significantly alter values and shape of ``data[field]``. The original data is
    kept in the data dios and assigned to the fieldname ``field + '_original'``.
    
    Note, that the data only gets interpolated at those (regular) timestamps, that have a valid (existing and
    not-na) datapoint preceeding them and one succeeding them within freq range.
    Regular timestamp that do not suffice this condition get nan assigned AND The associated flag will be of value
    ``flagger.UNFLAGGED``.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-regularized.
    freq : str
        An offset string. The frequency of the grid you want to interpolate your data at.
    method : {"linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
        "polynomial", "krogh", "piecewise_polynomial", "spline", "pchip", "akima"}: string
        The interpolation method you want to apply.
    order : int, default 1
        If your selected interpolation method can be performed at different *orders* - here you pass the desired
        order.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in `to_drop` from the interpolation process. Default results in ``flagger.BAD``
        values being dropped initially.
    """
    pass


def mapToOriginal(field, method, to_drop):
    """
    The Function function "undoes" regularization, by regaining the original data and projecting the
    flags calculated for the regularized data onto the original ones.
    
    Afterwards the regularized data is removed from the data dios and ``'field'`` will be associated
    with the original data "again".
    
    Wherever the flags in the original data are "better" then the regularized flags projected on them,
    they get overridden with this regularized flags value.
    
    Which regularized flags are to be projected on which original flags, is controlled by the "method" parameters.
    
    Generally, if you regularized with the method "X", you should pass the method "inverse_X" to the deharmonization.
    If you regularized with an interpolation, the method "inverse_interpolation" would be the appropriate choice.
    Also you should pass the same drop flags keyword.
    
    The deharm methods in detail:
    ("original_flags" are associated with the original data that is to be regained,
    "regularized_flags" are associated with the regularized data that is to be "deharmonized",
    "freq" refers to the regularized datas sampling frequencie)
    
    * ``'inverse_nagg'``: all original_flags within the range *+/- freq/2* of a regularized_flag, get assigned this
      regularized flags value. (if regularized_flags > original_flag)
    * ``'inverse_bagg'``: all original_flags succeeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_fagg'``: all original_flags preceeding a regularized_flag within the range of "freq", get assigned this
      regularized flags value. (if regularized_flag > original_flag)
    
    * ``'inverse_interpolation'``: all original_flags within the range *+/- freq* of a regularized_flag, get assigned this
      regularized flags value (if regularized_flag > original_flag).
    
    * ``'inverse_nshift'``: That original_flag within the range +/- *freq/2*, that is nearest to a regularized_flag,
      gets the regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_bshift'``: That original_flag succeeding a source flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if regularized_flag > original_flag)
    * ``'inverse_nshift'``: That original_flag preceeding a regularized flag within the range freq, that is nearest to a
      regularized_flag, gets assigned this regularized flags value. (if source_flag > original_flag)
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-deharmonized.
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift',
            'inverse_interpolation'}
        The method used for projection of regularized flags onto original flags. See description above for more
        details.
    to_drop : {List[str], str}, default None
        Flagtypes you want to drop before interpolation - effectively excluding values that are flagged
        with a flag in to_drop from the interpolation process. Default results in flagger.BAD
        values being dropped initially.
    """
    pass


def resample(field, freq, agg_func, max_invalid_total_d, max_invalid_consec_d, max_invalid_total_f, max_invalid_consec_f, flag_agg_func, empty_intervals_flag, to_drop, freq_check):
    """
    Function to resample the data. Afterwards the data will be sampled at regular (equidistant) timestamps
    (or Grid points). Sampling intervals therefor get aggregated with a function, specifyed by 'agg_func' parameter and
    the result gets projected onto the new timestamps with a method, specified by "method". The following method
    (keywords) are available:
    
    * ``'nagg'``: all values in the range (+/- `freq`/2) of a grid point get aggregated with agg_func and assigned to it.
    * ``'bagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the last
      grid point.
    * ``'fagg'``: all values in a sampling interval get aggregated with agg_func and the result gets assigned to the next
      grid point.
    
    
    Note, that. if possible, functions passed to agg_func will get projected internally onto pandas.resample methods,
    wich results in some reasonable performance boost - however, for this to work, you should pass functions that have
    the __name__ attribute initialised and the according methods name assigned to it.
    Furthermore, you shouldnt pass numpys nan-functions
    (``nansum``, ``nanmean``,...) because those for example, have ``__name__ == 'nansum'`` and they will thus not
    trigger ``resample.func()``, but the slower ``resample.apply(nanfunc)``. Also, internally, no nans get passed to
    the functions anyway, so that there is no point in passing the nan functions.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-resampled.
    freq : str
        An Offset String, that will be interpreted as the frequency you want to resample your data with.
    agg_func : Callable
        The function you want to use for aggregation.
    method: {'fagg', 'bagg', 'nagg'}, default 'bagg'
        Specifies which intervals to be aggregated for a certain timestamp. (preceding, succeeding or
        "surrounding" interval). See description above for more details.
    max_invalid_total_d : {None, int}, default None
        Maximum number of invalid (nan) datapoints, allowed per resampling interval. If max_invalid_total_d is
        exceeded, the interval gets resampled to nan. By default (``np.inf``), there is no bound to the number of nan
        values in an interval and only intervals containing ONLY nan values or those, containing no values at all,
        get projected onto nan
    max_invalid_consec_d : {None, int}, default None
        Maximum number of consecutive invalid (nan) data points, allowed per resampling interval.
        If max_invalid_consec_d is exceeded, the interval gets resampled to nan. By default (np.inf),
        there is no bound to the number of consecutive nan values in an interval and only intervals
        containing ONLY nan values, or those containing no values at all, get projected onto nan.
    max_invalid_total_f : {None, int}, default None
        Same as `max_invalid_total_d`, only applying for the flags. The flag regarded as "invalid" value,
        is the one passed to empty_intervals_flag (default=``flagger.BAD``).
        Also this is the flag assigned to invalid/empty intervals.
    max_invalid_consec_f : {None, int}, default None
        Same as `max_invalid_total_f`, only applying onto flags. The flag regarded as "invalid" value, is the one passed
        to empty_intervals_flag (default=flagger.BAD). Also this is the flag assigned to invalid/empty intervals.
    flag_agg_func : Callable, default: max
        The function you want to aggregate the flags with. It should be capable of operating on the flags dtype
        (usually ordered categorical).
    empty_intervals_flag : {None, str}, default None
        A Flag, that you want to assign to invalid intervals. Invalid are those intervals, that contain nan values only,
        or no values at all. Furthermore the empty_intervals_flag is the flag, serving as "invalid" identifyer when
        checking for `max_total_invalid_f` and `max_consec_invalid_f patterns`. Default triggers ``flagger.BAD`` to be
        assigned.
    to_drop : {None, str, List[str]}, default None
        Flags that refer to values you want to drop before resampling - effectively excluding values that are flagged
        with a flag in to_drop from the resampling process - this means that they also will not be counted in the
        the `max_consec`/`max_total evaluation`. `to_drop` = ``None`` results in NO flags being dropped initially.
    freq_check : {None, 'check', 'auto'}, default None
    
        * ``None``: do not validate frequency-string passed to `freq`
        * ``'check'``: estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
          if no uniform sampling rate could be estimated
        * ``'auto'``: estimate frequency and use estimate. (Ignores `freq` parameter.)
    """
    pass


def reindexFlags(field, method, source, freq, to_drop, freq_check):
    """
    The Function projects flags of "source" onto flags of "field". Wherever the "field" flags are "better" then the
    source flags projected on them, they get overridden with this associated source flag value.
    
    Which "field"-flags are to be projected on which source flags, is controlled by the "method" and "freq"
    parameters.
    
    method: (field_flag in associated with "field", source_flags associated with "source")
    
    'inverse_nagg' - all field_flags within the range +/- freq/2 of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    'inverse_bagg' - all field_flags succeeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    'inverse_fagg' - all field_flags preceeding a source_flag within the range of "freq", get assigned this source flags
        value. (if source_flag > field_flag)
    
    'inverse_interpolation' - all field_flags within the range +/- freq of a source_flag, get assigned this source flags value.
        (if source_flag > field_flag)
    
    'inverse_nshift' - That field_flag within the range +/- freq/2, that is nearest to a source_flag, gets the source
        flags value. (if source_flag > field_flag)
    'inverse_bshift' - That field_flag succeeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    'inverse_nshift' - That field_flag preceeding a source flag within the range freq, that is nearest to a
        source_flag, gets assigned this source flags value. (if source_flag > field_flag)
    
    'match' - any field_flag with a timestamp matching a source_flags timestamp gets this source_flags value
    (if source_flag > field_flag)
    
    Note, to undo or backtrack a resampling/shifting/interpolation that has been performed with a certain method,
    you can just pass the associated "inverse" method. Also you should pass the same drop flags keyword.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to project the source-flags onto.
    method : {'inverse_fagg', 'inverse_bagg', 'inverse_nagg', 'inverse_fshift', 'inverse_bshift', 'inverse_nshift'}
        The method used for projection of source flags onto field flags. See description above for more details.
    source : str
        The source source of flags projection.
    freq : {None, str},default None
        The freq determines the projection range for the projection method. See above description for more details.
        Defaultly (None), the sampling frequency of source is used.
    to_drop : {None, str, List[str]}, default None
        Flags referring to values that are to drop before flags projection. Relevant only when projecting with an
        inverted shift method. Defaultly flagger.BAD is listed.
    freq_check : {None, 'check', 'auto'}, default None
        - None: do not validate frequency-string passed to `freq`
        - 'check': estimate frequency and log a warning if estimate miss matchs frequency string passed to 'freq', or
            if no uniform sampling rate could be estimated
        - 'auto': estimate frequency and use estimate. (Ignores `freq` parameter.)
    """
    pass

