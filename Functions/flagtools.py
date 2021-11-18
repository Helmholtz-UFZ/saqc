"""

"""


def forceFlags(field, flag, kwargs):
    """
    Set whole column to a flag value.

    Parameters
    ----------
    field : str
        columns name that holds the data
    flag : float, default BAD
        flag to set
    kwargs : dict
        unused

    See Also
    --------
    clearFlags : set whole column to UNFLAGGED
    flagUnflagged : set flag value at all unflagged positions
    """
    pass


def clearFlags(field, kwargs):
    """
    Set whole column to UNFLAGGED.

    Parameters
    ----------
    field : str
        columns name that holds the data
    kwargs : dict
        unused

    Notes
    -----
    This function ignores the ``to_mask`` keyword, because the data is not relevant
    for processing.
    A warning is triggered if the ``flag`` keyword is given, because the flags are
    always set to `UNFLAGGED`.


    See Also
    --------
    forceFlags : set whole column to a flag value
    flagUnflagged : set flag value at all unflagged positions
    """
    pass


def flagUnflagged(field, flag, kwargs):
    """
    Function sets a flag at all unflagged positions.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flag : float, default BAD
        flag value to set
    kwargs : Dict
        unused

    Notes
    -----
    This function ignores the ``to_mask`` keyword, because the data is not relevant
    for processing.

    See Also
    --------
    clearFlags : set whole column to UNFLAGGED
    forceFlags : set whole column to a flag value
    """
    pass


def flagManual(field, mdata, method, mformat, mflag, flag):
    """
    Flag data by given, "manually generated" data.

    The data is flagged at locations where `mdata` is equal to a provided flag (`mflag`).
    The format of mdata can be an indexed object, like pd.Series, pd.Dataframe or dios.DictOfSeries,
    but also can be a plain list- or array-like.
    How indexed mdata is aligned to data is specified via the `method` parameter.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    mdata : pd.Series, pd.DataFrame, DictOfSeries, str, list or np.ndarray
        The Data determining, wich intervals are to be flagged, or a string, denoting under which field the data is
        accessable.
    method : {'plain', 'ontime', 'left-open', 'right-open', 'closed'}, default 'plain'
        Defines how mdata is projected on data. Except for the 'plain' method, the methods assume mdata to have an
        index.
        * 'plain': mdata must have the same length as data and is projected one-to-one on data.
        * 'ontime': works only with indexed mdata. mdata entries are matched with data entries that have the same index.
        * 'right-open': mdata defines intervals, values are to be projected on.
          The intervals are defined,
          (1) Either, by any two consecutive timestamps t_1 and 1_2 where t_1 is valued with mflag, or by a series,
          (2) Or, a Series, where the index contains in the t1 timestamps nd the values the respective t2 stamps.
          The value at t_1 gets projected onto all data timestamps t with t_1 <= t < t_2.
        * 'left-open': like 'right-open', but the projected interval now covers all t with t_1 < t <= t_2.
        * 'closed': like 'right-open', but the projected interval now covers all t with t_1 <= t <= t_2.
    mformat : {"start-end", "mflag"}, default "start-end"
        * "start-end": mdata is a Series, where every entry indicates an interval to-flag. The index defines the left
          bound, the value defines the right bound.
        * "mflag": mdata is an array like, with entries containing 'mflag',where flags shall be set. See documentation
          for examples.
    mflag : scalar
        The flag that indicates data points in `mdata`, of wich the projection in data should be flagged.
    flag : float, default BAD
        flag to set.

    Examples
    --------
    An example for mdata
    >>> mdata = pd.Series([1,0,1], index=pd.to_datetime(['2000-02', '2000-03', '2001-05']))
    >>> mdata
    2000-02-01    1
    2000-03-01    0
    2001-05-01    1
    dtype: int64

    On *dayly* data, with the 'ontime' method, only the provided timestamnps are used.
    Bear in mind that only exact timestamps apply, any offset will result in ignoring
    the timestamp.
    >>> _, fl = flagManual(data, field, flags, mdata, mflag=1, method='ontime')
    >>> fl[field] > UNFLAGGED
    2000-01-31    False
    2000-02-01    True
    2000-02-02    False
    2000-02-03    False
    ..            ..
    2000-02-29    False
    2000-03-01    True
    2000-03-02    False
    Freq: D, dtype: bool

    With the 'right-open' method, the mdata is forward fill:
    >>> _, fl = flagManual(data, field, flags, mdata, mflag=1, method='right-open')
    >>> fl[field] > UNFLAGGED
    2000-01-31    False
    2000-02-01    True
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    False
    2000-03-02    False
    Freq: D, dtype: bool

    With the 'left-open' method, backward filling is used:
    >>> _, fl = flagManual(data, field, flags, mdata, mflag=1, method='left-open')
    >>> fl[field] > UNFLAGGED
    2000-01-31    False
    2000-02-01    False
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    True
    2000-03-02    False
    Freq: D, dtype: bool
    """
    pass


def flagDummy(field):
    """
    Function does nothing but returning data and flags.

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass
