#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple
from typing_extensions import Literal

import numpy as np

from dios import DictOfSeries

from saqc.core.register import register
from saqc.flagger import Flagger
from saqc.lib.tools import periodicMask


@register(masking='none', module="tools")
def copy(data: DictOfSeries, field: str, flagger: Flagger, new_field: str, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    The function generates a copy of the data "field" and inserts it under the name field + suffix into the existing
    data.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to fork (copy).
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    new_field: str
        Target name.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        data shape may have changed relatively to the flagger input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags shape may have changed relatively to the flagger input.
    """

    if new_field in flagger.flags.columns.union(data.columns):
        raise ValueError(f"{field}: field already exist")

    flags, extras = flagger.getFlags(field, full=True)
    newflagger = flagger.replaceField(new_field, flags=flags, **extras)
    newdata = data.copy()
    newdata[new_field] = data[field].copy()
    return newdata, newflagger


@register(masking='none', module="tools")
def drop(data: DictOfSeries, field: str, flagger: Flagger, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    The function drops field from the data dios and the flagger.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to drop.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        data shape may have changed relatively to the flagger input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags shape may have changed relatively to the flagger input.
    """

    data = data.copy()
    del data[field]
    flagger = flagger.replaceField(field, flags=None)
    return data, flagger


@register(masking='none', module="tools")
def rename(data: DictOfSeries, field: str, flagger: Flagger, new_name: str, **kwargs) -> Tuple[DictOfSeries, Flagger]:
    """
    The function renames field to new name (in both, the flagger and the data).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the data column, you want to rename.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    new_name : str
        String, field is to be replaced with.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
    """
    # store
    s = data[field]
    f, e = flagger.getFlags(field, full=True)

    # delete
    data = data.copy()
    del data[field]
    flagger = flagger.replaceField(field, flags=None)

    # insert
    data[new_name] = s
    flagger = flagger.replaceField(new_name, inplace=True, flags=f, **e)

    return data, flagger


@register(masking='none', module="tools")
def mask(
        data: DictOfSeries,
        field: str,
        flagger: Flagger,
        mode: Literal["periodic", "mask_var"],
        mask_var: Optional[str]=None,
        period_start: Optional[str]=None,
        period_end: Optional[str]=None,
        include_bounds: bool=True
) -> Tuple[DictOfSeries, Flagger]:
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
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-masked.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
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

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        Data values may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.


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
    data = data.copy()
    datcol_idx = data[field].index

    if mode == 'periodic':
        to_mask = periodicMask(datcol_idx, period_start, period_end, include_bounds)
    elif mode == 'mask_var':
        idx = data[mask_var].index.intersection(datcol_idx)
        to_mask = data.loc[idx, mask_var]
    else:
        raise ValueError("Keyword passed as masking mode is unknown ({})!".format(mode))

    data.aloc[to_mask, field] = np.nan
    flagger = flagger.setFlags(field, loc=to_mask, flag=np.nan, force=True)

    return data, flagger
