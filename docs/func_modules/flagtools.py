"""

"""
def flagDummy(field):
    """
    Function does nothing but returning data and flagger.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass


def flagForceFail(field):
    """
    Function raises a runtime error.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass


def flagUnflagged(field, kwargs):
    """
    Function sets the flagger.GOOD flag to all values flagged better then flagger.GOOD.
    If there is an entry 'flag' in the kwargs dictionary passed, the
    function sets the kwargs['flag'] flag to all values flagged better kwargs['flag']
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    kwargs : Dict
        If kwargs contains 'flag' entry, kwargs['flag] is set, if no entry 'flag' is present,
        'flagger.UNFLAGGED' is set.
    """
    pass


def flagGood(field):
    """
    Function sets the flagger.GOOD flag to all values flagged better then flagger.GOOD.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass


def flagManual(field, mdata, mflag, method):
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
    mdata : {pd.Series, pd.Dataframe, DictOfSeries}
        The "manually generated" data
    mflag : scalar
        The flag that indicates data points in `mdata`, of wich the projection in data should be flagged.
    method : {'plain', 'ontime', 'left-open', 'right-open'}, default plain
        Defines how mdata is projected on data. Except for the 'plain' method, the methods assume mdata to have an
        index.
    
        * 'plain': mdata must have the same length as data and is projected one-to-one on data.
        * 'ontime': works only with indexed mdata. mdata entries are matched with data entries that have the same index.
        * 'right-open': mdata defines intervals, values are to be projected on.
          The intervals are defined by any two consecutive timestamps t_1 and 1_2 in mdata.
          the value at t_1 gets projected onto all data timestamps t with t_1 <= t < t_2.
        * 'left-open': like 'right-open', but the projected interval now covers all t with t_1 < t <= t_2.
    
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
    >>> _, fl = flagManual(data, field, flagger, mdata, mflag=1, method='ontime')
    >>> fl.isFlagged(field)
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
    >>> _, fl = flagManual(data, field, flagger, mdata, mflag=1, method='right-open')
    >>> fl.isFlagged(field)
    2000-01-31    False
    2000-02-01    True
    2000-02-02    True
    ..            ..
    2000-02-29    True
    2000-03-01    False
    2000-03-02    False
    Freq: D, dtype: bool
    
    With the 'left-open' method, backward filling is used:
    >>> _, fl = flagManual(data, field, flagger, mdata, mflag=1, method='left-open')
    >>> fl.isFlagged(field)
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

