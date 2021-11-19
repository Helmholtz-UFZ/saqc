"""

"""
def copyField(field):
    """
    The function generates a copy of the data "field" and inserts it under the name field + suffix into the existing
    data.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to fork (copy).
    """
    pass


def dropField(field):
    """
    The function drops field from the data dios and the flags.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to drop.
    """
    pass


def renameField(field, new_name):
    """
    The function renames field to new name (in both, the flags and the data).
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to rename.
    new_name : str
        String, field is to be replaced with.
    """
    pass


def maskTime(field, mode, mask_field, start, end, closed):
    """
    This function realizes masking within saqc.
    
    Due to some inner saqc mechanics, it is not straight forwardly possible to exclude
    values or datachunks from flagging routines. This function replaces flags with UNFLAGGED
    value, wherever values are to get masked. Furthermore, the masked values get replaced by
    np.nan, so that they dont effect calculations.
    
    Here comes a recipe on how to apply a flagging function only on a masked chunk of the variable field:
    
    1. dublicate "field" in the input data (copy)
    2. mask the dublicated data (mask)
    3. apply the tests you only want to be applied onto the masked data chunks (saqc_tests)
    4. project the flags, calculated on the dublicated and masked data onto the original field data
        (projectFlags or flagGeneric)
    5. drop the dublicated data (drop)
    
    To see an implemented example, checkout flagSeasonalRange in the saqc.functions module
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-masked.
    mode : {"periodic", "mask_var"}
        The masking mode.
        - "periodic": parameters "period_start", "end" are evaluated to generate a periodical mask
        - "mask_var": data[mask_var] is expected to be a boolean valued timeseries and is used as mask.
    mask_field : {None, str}, default None
        Only effective if mode == "mask_var"
        Fieldname of the column, holding the data that is to be used as mask. (must be boolean series)
        Neither the series` length nor its labels have to match data[field]`s index and length. An inner join of the
        indices will be calculated and values get masked where the values of the inner join are ``True``.
    start : {None, str}, default None
        Only effective if mode == "seasonal"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `end` parameter.
        See examples section below for some examples.
    end : {None, str}, default None
        Only effective if mode == "periodic"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `end` parameter.
        See examples section below for some examples.
    closed : boolean
        Wheather or not to include the mask defining bounds to the mask.
    
    Examples
    --------
    The `period_start` and `end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
    They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
    (mm=month, dd=day, HH=hour, MM=minute, SS=second)
    Single digit specifications have to be given with leading zeros.
    `period_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
    The highest date unit gives the period.
    For example:
    
    >>> period_start = "01T15:00:00"
    >>> end = "13T17:30:00"
    
    Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked
    
    >>> period_start = "01:00"
    >>> end = "04:00"
    
    All the values between the first and 4th minute of every hour get masked.
    
    >>> period_start = "01-01T00:00:00"
    >>> end = "01-03T00:00:00"
    
    Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
    include 00:00:00 at the first of march. To exclude this one, pass:
    
    >>> period_start = "01-01T00:00:00"
    >>> end = "02-28T23:59:59"
    
    To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
    season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:
    
    >>> period_start = "22:00:00"
    >>> end = "06:00:00"
    
    When inclusive_selection="season", all above examples work the same way, only that you now
    determine wich values NOT TO mask (=wich values are to constitute the "seasons").
    """
    pass


def plot(field, path, max_gap, stats, history, xscope, phaseplot, store_kwargs):
    """
    Stores or shows a figure object, containing data graph with flag marks for field.
    
    There are two modes, 'interactive' and 'store', which are determind through the
    ``save_path`` keyword. In interactive mode (default) the plot is shown at runtime
    and the program execution stops until the plot window is closed manually. In
    store mode the generated plot is stored to disk and no manually interaction is
    needed.
    
    Parameters
    ----------
    field : str
        Name of the variable-to-plot
    
    path : str, default None
        If ``None`` is passed, interactive mode is entered; plots are shown immediatly
        and a user need to close them manually before execution continues.
        If a filepath is passed instead, store-mode is entered and
        the plot is stored unter the passed location.
    
    max_gap : str, default None
        If None, all the points in the data will be connected, resulting in long linear
        lines, where continous chunks of data is missing. Nans in the data get dropped
        before plotting. If an offset string is passed, only points that have a distance
        below `max_gap` get connected via the plotting line.
    
    stats : bool, default False
        Whether to include statistics table in plot.
    
    history : {"valid", "complete", None}, default "valid"
        Discriminate the plotted flags with respect to the tests they originate from.
        * "valid" - Only plot those flags, that do not get altered or "unflagged" by subsequent tests. Only list tests
          in the legend, that actually contributed flags to the overall resault.
        * "complete" - plot all the flags set and list all the tests ran on a variable. Suitable for debugging/tracking.
        * "clear" - clear plot from all the flagged values
        * None - just plot the resulting flags for one variable, without any historical meta information.
    
    xscope : slice or Offset, default None
        Parameter, that determines a chunk of the data to be plotted
        processed. `xscope` can be anything, that is a valid argument to the ``pandas.Series.__getitem__`` method.
    
    phaseplot : str or None, default None
        If a string is passed, plot ``field`` in the phase space it forms together with the Variable ``phaseplot``.
    
    
    store_kwargs : dict, default {}
        Keywords to be passed on to the ``matplotlib.pyplot.savefig`` method, handling
        the figure storing. To store an pickle object of the figure, use the option
        ``{'pickle': True}``, but note that all other store_kwargs are ignored then.
        Reopen with: ``pickle.load(open(savepath,'w')).show()``
    
    stats_dict: dict, default None
        (Only relevant if `stats`=True)
        Dictionary of additional statisticts to write to the statistics table
        accompanying the data plot. An entry to the stats_dict has to be of the form:
    
        * {"stat_name": lambda x, y, z: func(x, y, z)}
    
        The lambda args ``x``,``y``,``z`` will be fed by:
    
        * ``x``: the data (``data[field]``).
        * ``y``: the flags (``flags[field]``).
        * ``z``: The passed flags level (``kwargs[flag]``)
    
        See examples section for examples
    
    Examples
    --------
    Summary statistic function examples:
    
    >>> func = lambda x, y, z: len(x)
    
    Total number of nan-values:
    
    >>> func = lambda x, y, z: x.isna().sum()
    
    Percentage of values, flagged greater than passed flag (always round float results
    to avoid table cell overflow):
    
    >>> func = lambda x, y, z: round((x.isna().sum()) / len(x), 2)
    """
    pass

