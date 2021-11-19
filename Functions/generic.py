"""

"""


def genericProcess(field, func):
    """
    Generate/process data with user defined functions.

    Formally, what the function does, is the following:

    1.  Let F be a Callable, depending on fields f_1, f_2,...f_K, (F = F(f_1, f_2,...f_K))
        Than, for every timestamp t_i that occurs in at least one of the timeseries data[f_j] (outer join),
        The value v_i is computed via:
        v_i = data([f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]), if all data[f_j][t_i] do exist
        v_i = ``np.nan``, if at least one of the data[f_j][t_i] is missing.
    2.  The result is stored to ``data[target]``, if ``target`` is given or to ``data[field]`` otherwise

    Parameters
    ----------
    field : str or list of str
        The variable(s) passed to func.
    func : callable
        Function to call on the variables given in ``field``. The return value will be written
        to ``target`` or ``field`` if the former is not given. This implies, that the function
        needs to accept the same number of arguments (of type pandas.Series) as variables given
        in ``field`` and should return an iterable of array-like objects with the same number
        of elements as given in ``target`` (or ``field`` if ``target`` is not specified).
    target: str or list of str
        The variable(s) to write the result of ``func`` to. If not given, the variable(s)
        specified in ``field`` will be overwritten. If a ``target`` is not given, it will be
        created.
    flag: float, default ``UNFLAGGED``
        The quality flag to set. The default ``UNFLAGGED`` states the general idea, that
        ``genericProcess`` generates 'new' data without direct relation to the potentially
        already present flags.
    to_mask: float, default ``UNFLAGGED``
        Threshold flag. Flag values greater than ``to_mask`` indicate that the associated
        data value is inappropiate for further usage.

    Examples
    --------
    Compute the sum of the variables 'rainfall' and 'snowfall' and save the result to
    a (new) variable 'precipitation'

    .. testsetup:: *

       saqc = saqc.SaQC(pd.DataFrame({'rainfall':[], 'snowfall':[]}))

    >>> saqc.genericProcess(field=["rainfall", "snowfall"], target="precipitation'", func=lambda x, y: x + y)
    """
    pass


def genericFlag(field, func):
    """
    Flag data with user defined functions.

    Formally, what the function does, is the following:
    Let X be a Callable, depending on fields f_1, f_2,...f_K, (X = X(f_1, f_2,...f_K))
    Than for every timestamp t_i in data[field]:
    data[field][t_i] is flagged if X(data[f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]) is True.

    Parameters
    ----------
    field : str or list of str
        The variable(s) passed to func.
    func : callable
        Function to call on the variables given in ``field``. The function needs to accept the same
        number of arguments (of type pandas.Series) as variables given in ``field`` and return an
        iterable of array-like objects of with dtype bool and with the same number of elements as
        given in ``target`` (or ``field`` if ``target`` is not specified). The function output
        determines the values to flag.
    target: str or list of str
        The variable(s) to write the result of ``func`` to. If not given, the variable(s)
        specified in ``field`` will be overwritten. If a ``target`` is not given, it will be
        created.
    flag: float, default ``UNFLAGGED``
        The quality flag to set. The default ``UNFLAGGED`` states the general idea, that
        ``genericProcess`` generates 'new' data without direct relation to the potentially
        already present flags.
    to_mask: float, default ``UNFLAGGED``
        Threshold flag. Flag values greater than ``to_mask`` indicate that the associated
        data value is inappropiate for further usage.

    Examples
    --------

    .. testsetup:: *

       saqc = saqc.SaQC(pd.DataFrame({'temperature':[0], 'uncertainty':[0], 'rainfall':[0], 'fan':[0]}))

    1. Flag the variable 'rainfall', if the sum of the variables 'temperature' and 'uncertainty' is below zero:

    >>> saqc.genericFlag(field=["temperature", "uncertainty"], target="rainfall", func= lambda x, y: temperature + uncertainty < 0

    2. Flag the variable 'temperature', where the variable 'fan' is flagged:

    >>> saqc.genericFlag(field="fan", target="temperature", func=lambda x: isflagged(x))

    3. The generic functions also support all pandas and numpy functions:

    >>> saqc.genericFlag(field="fan", target="temperature", func=lambda x: np.sqrt(x) < 7)
    """
    pass
