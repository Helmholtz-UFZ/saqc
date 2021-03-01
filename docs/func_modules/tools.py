"""

"""
def copy(field):
    """
    The function generates a copy of the data "field" and inserts it under the name field + suffix into the existing
    data.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to fork (copy).
    """
    pass


def drop(field):
    """
    The function drops field from the data dios and the flagger.
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to drop.
    """
    pass


def rename(field, new_name):
    """
    The function renames field to new name (in both, the flagger and the data).
    
    Parameters
    ----------
    field : str
        The fieldname of the data column, you want to rename.
    new_name : str
        String, field is to be replaced with.
    """
    pass


def mask(field, mode, mask_var, period_start, period_end, include_bounds):
    """
    This function realizes masking within saqc.
    
    Due to some inner saqc mechanics, it is not straight forwardly possible to exclude
    values or datachunks from flagging routines. This function replaces flags with np.nan
    value, wherever values are to get masked. Furthermore, the masked values get replaced by
    np.nan, so that they dont effect calculations.
    
    Here comes a recipe on how to apply a flagging function only on a masked chunk of the variable field:
    
    1. dublicate "field" in the input data (proc_copy)
    2. mask the dublicated data (modelling_mask)
    3. apply the tests you only want to be applied onto the masked data chunks (saqc_tests)
    4. project the flags, calculated on the dublicated and masked data onto the original field data
        (proc_projectFlags or flagGeneric)
    5. drop the dublicated data (proc_drop)
    
    To see an implemented example, checkout flagSeasonalRange in the saqc.functions module
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-masked.
    mode : {"periodic", "mask_var"}
        The masking mode.
        - "periodic": parameters "period_start", "period_end" are evaluated to generate a periodical mask
        - "mask_var": data[mask_var] is expected to be a boolean valued timeseries and is used as mask.
    mask_var : {None, str}, default None
        Only effective if mode == "mask_var"
        Fieldname of the column, holding the data that is to be used as mask. (must be moolean series)
        Neither the series` length nor its labels have to match data[field]`s index and length. An inner join of the
        indices will be calculated and values get masked where the values of the inner join are "True".
    period_start : {None, str}, default None
        Only effective if mode == "seasonal"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `period_end` parameter.
        See examples section below for some examples.
    period_end : {None, str}, default None
        Only effective if mode == "periodic"
        String denoting starting point of every period. Formally, it has to be a truncated instance of "mm-ddTHH:MM:SS".
        Has to be of same length as `period_end` parameter.
        See examples section below for some examples.
    include_bounds : boolean
        Wheather or not to include the mask defining bounds to the mask.
    
    Examples
    --------
    The `period_start` and `period_end` parameters provide a conveniant way to generate seasonal / date-periodic masks.
    They have to be strings of the forms: "mm-ddTHH:MM:SS", "ddTHH:MM:SS" , "HH:MM:SS", "MM:SS" or "SS"
    (mm=month, dd=day, HH=hour, MM=minute, SS=second)
    Single digit specifications have to be given with leading zeros.
    `period_start` and `seas   on_end` strings have to be of same length (refer to the same periodicity)
    The highest date unit gives the period.
    For example:
    
    >>> period_start = "01T15:00:00"
    >>> period_end = "13T17:30:00"
    
    Will result in all values sampled between 15:00 at the first and  17:30 at the 13th of every month get masked
    
    >>> period_start = "01:00"
    >>> period_end = "04:00"
    
    All the values between the first and 4th minute of every hour get masked.
    
    >>> period_start = "01-01T00:00:00"
    >>> period_end = "01-03T00:00:00"
    
    Mask january and february of evcomprosed in theery year. masking is inclusive always, so in this case the mask will
    include 00:00:00 at the first of march. To exclude this one, pass:
    
    >>> period_start = "01-01T00:00:00"
    >>> period_end = "02-28T23:59:59"
    
    To mask intervals that lap over a seasons frame, like nights, or winter, exchange sequence of season start and
    season end. For example, to mask night hours between 22:00:00 in the evening and 06:00:00 in the morning, pass:
    
    >>> period_start = "22:00:00"
    >>> period_end = "06:00:00"
    
    When inclusive_selection="season", all above examples work the same way, only that you now
    determine wich values NOT TO mask (=wich values are to constitute the "seasons").
    """
    pass

