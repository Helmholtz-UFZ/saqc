#! /usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from inspect import signature
from typing import Tuple, Any, Union, Callable

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.common import *
from saqc.core.register import register
from saqc.core.visitor import ENVIRONMENT
from saqc.flagger import Flagger, initFlagsLike

import operator as op

_OP = {'<': op.lt, '<=': op.le, '==': op.eq, '!=': op.ne, '>': op.gt, '>=': op.ge}


def _dslIsFlagged(
        flagger: Flagger, var: pd.Series, flag: float = UNFLAGGED, comparator: str = ">"
) -> Union[pd.Series, DictOfSeries]:
    """
    helper function for `flag`
    """
    comparison = _OP[comparator]
    return comparison(flagger[var.name], flag)


def _execGeneric(flagger: Flagger, data: DictOfSeries, func: Callable[[pd.Series], pd.Series], field: str,
                 nodata: float) -> pd.Series:
    # TODO:
    # - check series.index compatibility
    # - field is only needed to translate 'this' parameters
    #    -> maybe we could do the translation on the tree instead

    sig = signature(func)
    args = []
    for k, v in sig.parameters.items():
        k = field if k == "this" else k
        if k not in data:
            raise NameError(f"variable '{k}' not found")
        args.append(data[k])

    globs = {
        "isflagged": partial(_dslIsFlagged, flagger),
        "ismissing": lambda var: ((var == nodata) | pd.isnull(var)),
        "mask": lambda cond: data[cond.name].mask(cond),
        "this": field,
        "NODATA": nodata,
        "GOOD": GOOD,
        "BAD": BAD,
        "UNFLAGGED": UNFLAGGED,
        **ENVIRONMENT,
    }
    func.__globals__.update(globs)
    return func(*args)


@register(masking='all', module="generic")
def process(data: DictOfSeries, field: str, flagger: Flagger, func: Callable[[pd.Series], pd.Series],
            nodata: float = np.nan, **kwargs) -> Tuple[DictOfSeries, Flagger]:
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
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, where you want the result from the generic expressions processing to be written to.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    func : Callable
        The data processing function with parameter names that will be
        interpreted as data column entries.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        The shape of the data may have changed relatively to the data input.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        The flags shape may have changed relatively to the input flagger.

    Examples
    --------
    Some examples on what to pass to the func parameter:
    To compute the sum of the variables "temperature" and "uncertainty", you would pass the function:

    >>> lambda temperature, uncertainty: temperature + uncertainty

    You also can pass numpy and pandas functions:

    >>> lambda temperature, uncertainty: np.round(temperature) * np.sqrt(uncertainty)

    """
    data[field] = _execGeneric(flagger, data, func, field, nodata).squeeze()

    # todo: the former comment wished to overwrite the column, but i'm not sure -- palmb
    if field in flagger:
        flagger.drop(field)

    flagger[field] = initFlagsLike(data[field])[field]
    return data, flagger


@register(masking='all', module="generic")
def flag(data: DictOfSeries, field: str, flagger: Flagger, func: Callable[[pd.Series], pd.Series],
         nodata: float = np.nan, flag=BAD, **kwargs) -> Tuple[DictOfSeries, Flagger]:
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
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, where you want the result from the generic expressions evaluation to be projected
        to.
    flagger : saqc.flagger.Flagger
        A flagger object, holding flags and additional Informations related to `data`.
    func : Callable
        The expression that is to be evaluated is passed in form of a callable, with parameter names that will be
        interpreted as data column entries. The Callable must return an boolen array like.
        See the examples section to learn more.
    nodata : any, default np.nan
        The value that indicates missing/invalid data

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.Flagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

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
    'temperature', if 'level' is flagged at a level named DOUBTFUL or worse, use:

    >>> lambda level: isflagged(level, flag=DOUBTFUL, comparator='>')

    # TODO : fix text
    If you are unsure about the used flaggers flagging level names, you can use the reserved key words BAD, UNFLAGGED
    and GOOD, to refer to the worst (BAD), best(GOOD) or unflagged (UNFLAGGED) flagging levels. For example.

    >>> lambda level: isflagged(level, flag=UNFLAGGED, comparator='==')

    Your expression also is allowed to include pandas and numpy functions

    >>> lambda level: np.sqrt(level) > 7
    """
    # NOTE:
    # The naming of the func parameter is pretty confusing
    # as it actually holds the result of a generic expression
    mask = _execGeneric(flagger, data, func, field, nodata).squeeze()
    if np.isscalar(mask):
        raise TypeError(f"generic expression does not return an array")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"generic expression does not return a boolean array")

    if field not in flagger.columns:
        flagger[field] = pd.Series(UNFLAGGED, index=mask.index, name=field)

    # if flagger.getFlags(field).empty:
    #     flagger = flagger.merge(
    #         flagger.initFlags(
    #             data=pd.Series(name=field, index=mask.index, dtype=np.float64)))
    flagger[mask, field] = flag
    return data, flagger
