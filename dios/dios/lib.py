import pandas as pd
import warnings


class ItypeWarning(RuntimeWarning):
    pass


class ItypeCastWarning(ItypeWarning):
    pass


class ItypeCastError(RuntimeError):
    pass


class __Itype:
    def __init__(self):
        raise RuntimeError("a Itype class does not allow instances of itself.")


class DtItype(__Itype):
    name = "datetime"
    unique = True
    subtypes = (pd.DatetimeIndex,)
    min_pdindex = pd.DatetimeIndex([])


class IntItype(__Itype):
    name = "integer"
    unique = True
    subtypes = (pd.RangeIndex, pd.Int64Index, pd.UInt64Index, int)
    min_pdindex = pd.Int64Index([])


class FloatItype(__Itype):
    name = "float"
    subtypes = (pd.Float64Index, float)
    unique = True
    min_pdindex = pd.Float64Index([])


# class MultiItype(__Itype):
#     name = "multi"
#     subtypes = (pd.MultiIndex, )
#     unique = ??


class NumItype(__Itype):
    name = "numeric"
    _subitypes = (IntItype, FloatItype)
    subtypes = _subitypes + IntItype.subtypes + FloatItype.subtypes
    unique = False
    min_pdindex = pd.Float64Index([])


class ObjItype(__Itype):
    name = "object"
    unique = False
    _subitypes = (DtItype, IntItype, FloatItype, NumItype, str)
    _otheritypes = (
        pd.CategoricalIndex,
        pd.IntervalIndex,
        pd.PeriodIndex,
        pd.TimedeltaIndex,
        pd.Index,
    )
    subtypes = _subitypes + _otheritypes + DtItype.subtypes + NumItype.subtypes
    min_pdindex = pd.Index([])


def is_itype(obj, itype):
    """Check if obj is a instance of the given itype or its str-alias was given"""

    # todo: iter through itype as it could be a tuple, if called like ``is_itype(o, (t1,t2))``

    # user gave a Itype, like ``DtItype``
    if type(obj) == type and issubclass(obj, itype):
        return True

    # user gave a string, like 'datetime'
    if isinstance(obj, str) and obj == itype.name:
        return True

    return False


def is_itype_subtype(obj, itype):
    """Check if obj is a subclass or a instance of a subclass of the given itype"""

    # user gave a subtype, like ``pd.DatetimeIndex``
    if type(obj) == type and issubclass(obj, itype.subtypes):
        return True

    # user gave a instance of a subtype, like ``pd.Series(..).index``
    if isinstance(obj, itype.subtypes):
        return True

    return False


def is_itype_like(obj, itype):
    """Check if obj is a subclass or a instance of the given itype or any of its subtypes"""
    return is_itype(obj, itype) or is_itype_subtype(obj, itype)


def get_itype(obj):
    """

    Return the according Itype.

    and return the according Itype
    Parameters
    ----------
    obj : {itype string, Itype, pandas.Index, instance of pd.index}
        get the itype fitting for the input

    Examples
    --------
    >>> get_itype("datetime")
    <class 'dios.lib.DtItype'>

    >>> s = pd.Series(index=pd.to_datetime([]))
    >>> get_itype(s.index)
    <class 'dios.lib.DtItype'>

    >>> get_itype(DtItype)
    <class 'dios.lib.DtItype'>

    >>> get_itype(pd.DatetimeIndex)
    <class 'dios.lib.DtItype'>
    """
    if type(obj) == type and issubclass(obj, __Itype):
        return obj

    # check if it is the actual type, not a subtype
    types = [DtItype, IntItype, FloatItype, NumItype, ObjItype]
    for t in types:
        if is_itype(obj, t):
            return t

    for t in types:
        if is_itype_subtype(obj, t):
            return t

    raise ValueError(
        f"{obj} is not a itype, nor any known subtype of a itype, nor a itype string alias"
    )


def _itype_eq(a, b):
    return is_itype(a, b)


def _itype_lt(a, b):
    return is_itype_subtype(a, b)


def _itype_le(a, b):
    return is_itype_like(a, b)


def _find_least_common_itype(iterable_of_series):
    itypes = [NumItype, FloatItype, IntItype, DtItype]
    tlist = [get_itype(s.index) for s in iterable_of_series]
    found = ObjItype
    if tlist:
        for itype in itypes:
            for t in tlist:
                if _itype_le(t, itype):
                    continue
                break
            else:
                found = itype
    return found


################################################################################
# Casting


class CastPolicy:
    force = "force"
    save = "save"
    never = "never"


_CAST_POLICIES = [CastPolicy.force, CastPolicy.save, CastPolicy.never]


def cast_to_itype(series, itype, policy="lossless", err="raise", inplace=False):
    """Cast a series (more explicit the type of the index) to fit the itype of a dios.

    Return the casted series if successful, None otherwise.

    Note:
        This is very basic number-casting, so in most cases, information from
        the old index will be lost after the cast.
    """

    if policy not in _CAST_POLICIES:
        raise ValueError(f"policy={policy}")
    if err not in ["raise", "ignore"]:
        raise ValueError(f"err={err}")
    if not inplace:
        series = series.copy()
    itype = get_itype(itype)

    if series.empty:
        return pd.Series(index=itype.min_pdindex, dtype=series.dtype)

    series.itype = get_itype(series.index)

    # up-cast issn't necessary because a dios with a higher
    # itype always can take lower itypes.
    # series can have dt/int/float/mixed
    # dt    -> dt           -> mixed
    # int   -> int   -> num -> mixed
    # float -> float -> num -> mixed
    # mixed                 -> mixed
    if _itype_le(series.itype, itype):  # a <= b
        return series

    e = f"A series index of type '{type(series.index)}' cannot be casted to Itype '{itype.name}'"

    # cast any -> dt always fail.
    if is_itype(itype, DtItype):
        pass
    else:
        e += f", as forbidden by the cast-policy '{policy}'."

    if policy == CastPolicy.never:
        pass

    elif policy == CastPolicy.force:
        # cast any (dt/float/mixed) -> int
        if is_itype(itype, IntItype):  # a == b
            series.index = pd.RangeIndex(len(series))
            return series
        # cast any (dt/int/mixed) -> float
        # cast any (dt/float/mixed) -> nur
        if is_itype(itype, FloatItype) or is_itype(itype, NumItype):  # a == b or a == c
            series.index = pd.Float64Index(range(len(series)))
            return series

    elif policy == CastPolicy.save:
        # cast int   -> float
        if is_itype(itype, IntItype) and is_itype(
            series.itype, FloatItype
        ):  # a == b and c == d
            series.index = series.index.astype(float)
            return series
        # cast float -> int, maybe if unique
        if is_itype(itype, FloatItype) and is_itype(
            series.itype, IntItype
        ):  # a == b and c == d
            series.index = series.index.astype(int)
            if series.index.is_unique:
                return series
            e = (
                f"The cast with policy {policy} from series index type '{type(series.index)}' to "
                f"itype {itype.name} resulted in a non-unique index."
            )
        # cast mixed -> int/float always fail

    if err == "raise":
        raise ItypeCastError(e)
    else:
        return None


################################################################################
# OPTIONS


class OptsFields:
    """storage class for the keys in `dios_options`

    Use like so: ``dios_options[OptsFields.X] = Opts.Y``.

    See Also
    --------
        Opts: values for the options dict
        dios_options: options dict for module
    """

    mixed_itype_warn_policy = "mixed_itype_policy"
    disp_max_rows = "disp_max_rows "
    disp_min_rows = "disp_min_rows "
    disp_max_cols = "disp_max_vars"
    dios_repr = "dios_repr"


class Opts:
    """storage class for string values for `dios_options`

    Use like so: ``dios_options[OptsFields.X] = Opts.Y``.

    See Also
    --------
        OptsFields: keys for the options dict
        dios_options: options dict for module
    """

    itype_warn = "warn"
    itype_err = "err"
    itype_ignore = "ignore"
    repr_aligned = "aligned"
    repr_indexed = "indexed"


class __DocDummy(dict):
    pass


dios_options = __DocDummy()
dios_options.update(
    **{
        OptsFields.disp_max_rows: 60,
        OptsFields.disp_min_rows: 10,
        OptsFields.disp_max_cols: 10,
        OptsFields.mixed_itype_warn_policy: Opts.itype_warn,
        OptsFields.dios_repr: Opts.repr_indexed,
    }
)

opdoc = f"""Options dictionary for module `dios`.

Use like so: ``dios_options[OptsFields.X] = Opts.Y``.

**Items**:
 * {OptsFields.dios_repr}: {{'indexed', 'aligned'}} default: 'indexed'
    dios default representation if:
     * `indexed`:  show every column with its index
     * `aligned`:  transform to pandas.DataFrame with indexed merged together.
 * {OptsFields.disp_max_rows}  : int
    Maximum numbers of row before truncated to `disp_min_rows`
    in representation of DictOfSeries

 * {OptsFields.disp_min_rows} : int
    min rows to display if `max_rows` is exceeded

 * {OptsFields.disp_max_cols} : int
    Maximum numbers of columns before truncated representation

 * {OptsFields.mixed_itype_warn_policy} : {{'warn', 'err', 'ignore'}}
    How to inform user about mixed Itype

See Also
--------
    OptsFields: keys for the options dict 
    Opts: values for the options dict 

"""
dios_options.__doc__ = opdoc


def _throw_MixedItype_err_or_warn(itype):
    msg = (
        f"Using '{itype.name}' as itype is not recommend. "
        f"As soon as series with different index types are inserted,\n"
        f"indexing and slicing will almost always fail. "
    )

    if dios_options[OptsFields.mixed_itype_warn_policy] in [
        "ignore",
        Opts.itype_ignore,
    ]:
        pass
    elif dios_options[OptsFields.mixed_itype_warn_policy] in [
        "error",
        "err",
        Opts.itype_err,
    ]:
        msg += "Suppress this error by specifying an unitary 'itype' or giving an 'index' to DictOfSeries."
        raise ItypeCastError(msg)
    else:
        msg += "Silence this warning by specifying an unitary 'itype' or giving an 'index' to DictOfSeries."
        warnings.warn(msg, ItypeWarning)
    return


def example_DictOfSeries():
    """Return a example dios.

    Returns
    -------
    DictOfSeries: an example

    Examples
    --------

    >>> from dios import example_DictOfSeries
    >>> di = example_DictOfSeries()
    >>> di
        a |      b |      c |     d |
    ===== | ====== | ====== | ===== |
    0   0 | 2    5 | 4    7 | 6   0 |
    1   7 | 3    6 | 5   17 | 7   1 |
    2  14 | 4    7 | 6   27 | 8   2 |
    3  21 | 5    8 | 7   37 | 9   3 |
    4  28 | 6    9 | 8   47 | 10  4 |
    5  35 | 7   10 | 9   57 | 11  5 |
    6  42 | 8   11 | 10  67 | 12  6 |
    7  49 | 9   12 | 11  77 | 13  7 |
    8  56 | 10  13 | 12  87 | 14  8 |
    9  63 | 11  14 | 13  97 | 15  9 |
    """
    from dios import DictOfSeries

    a = pd.Series(range(0, 70, 7))
    b = pd.Series(range(5, 15, 1))
    c = pd.Series(range(7, 107, 10))
    d = pd.Series(range(0, 10, 1))

    for i, s in enumerate([a, b, c, d]):
        s.index += i * 2

    di = DictOfSeries(dict(a=a, b=b, c=c, d=d))
    return di.copy()
