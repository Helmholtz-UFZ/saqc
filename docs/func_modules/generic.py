"""

"""
def process(field, func, nodata):
    """
    generate/process data with generically defined functions.
    
    The functions can depend on on any of the fields present in data.
    
    Formally, what the function does, is the following:
    
    1.  Let F be a Callable, depending on fields f_1, f_2,...f_K, (F = F(f_1, f_2,...f_K))
        Than, for every timestamp t_i that occurs in at least one of the timeseries data[f_j] (outer join),
        The value v_i is computed via:
        v_i = data([f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]), if all data[f_j][t_i] do exist
        v_i = `nodata`, if at least one of the data[f_j][t_i] is missing.
    2.  The result is stored to data[field] (gets generated if not present)
    
    Parameters
    ----------
    field : str
        The fieldname of the column, where you want the result from the generic expressions processing to be written to.
    func : Callable
        The data processing function with parameter names that will be
        interpreted as data column entries.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data
    
    Examples
    --------
    Some examples on what to pass to the func parameter:
    To compute the sum of the variables "temperature" and "uncertainty", you would pass the function:
    
    >>> lambda temperature, uncertainty: temperature + uncertainty
    
    You also can pass numpy and pandas functions:
    
    >>> lambda temperature, uncertainty: np.round(temperature) * np.sqrt(uncertainty)
    """
    pass


def flag(field, func, nodata):
    """
    a function to flag a data column by evaluation of a generic expression.
    
    The expression can depend on any of the fields present in data.
    
    Formally, what the function does, is the following:
    
    Let X be an expression, depending on fields f_1, f_2,...f_K, (X = X(f_1, f_2,...f_K))
    Than for every timestamp t_i in data[field]:
    data[field][t_i] is flagged if X(data[f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]) is True.
    
    Note, that all value series included in the expression to evaluate must be labeled identically to field.
    
    Note, that the expression is passed in the form of a Callable and that this callables variable names are
    interpreted as actual names in the data header. See the examples section to get an idea.
    
    Note, that all the numpy functions are available within the generic expressions.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, where you want the result from the generic expressions evaluation to be projected
        to.
    func : Callable
        The expression that is to be evaluated is passed in form of a callable, with parameter names that will be
        interpreted as data column entries. The Callable must return an boolen array like.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data
    
    Examples
    --------
    Some examples on what to pass to the func parameter:
    To flag the variable `field`, if the sum of the variables
    "temperature" and "uncertainty" is below zero, you would pass the function:
    
    >>> lambda temperature, uncertainty: temperature + uncertainty < 0
    
    There is the reserved name 'This', that always refers to `field`. So, to flag field if field is negative, you can
    also pass:
    
    >>> lambda this: this < 0
    
    If you want to make dependent the flagging from flags already present in the data, you can use the built-in
    ``isflagged`` method. For example, to flag the 'temperature', if 'level' is flagged, you would use:
    
    >>> lambda level: isflagged(level)
    
    You can furthermore specify a flagging level, you want to compare the flags to. For example, for flagging
    'temperature', if 'level' is flagged at a level named 'doubtfull' or worse, use:
    
    >>> lambda level: isflagged(level, flag='doubtfull', comparator='<=')
    
    If you are unsure about the used flaggers flagging level names, you can use the reserved key words BAD, UNFLAGGED
    and GOOD, to refer to the worst (BAD), best(GOOD) or unflagged (UNFLAGGED) flagging levels. For example.
    
    >>> lambda level: isflagged(level, flag=UNFLAGGED, comparator='==')
    
    Your expression also is allowed to include pandas and numpy functions
    
    >>> lambda level: np.sqrt(level) > 7
    """
    pass

