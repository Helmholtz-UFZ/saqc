#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.constants import BAD, UNFLAGGED, ENVIRONMENT
from saqc.lib.tools import toSequence
from saqc.lib.types import GenericFunction, PandasLike
from saqc.core.flags import Flags
from saqc.core.register import register, _isflagged


def _execGeneric(
    flags: Flags,
    data: PandasLike,
    func: GenericFunction,
    to_mask: float = UNFLAGGED,
) -> DictOfSeries:

    globs = {
        "isflagged": lambda data: _isflagged(flags[data.name], thresh=to_mask),
        **ENVIRONMENT,
    }

    func.__globals__.update(globs)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    out = func(*[data[c] for c in data.columns])
    return DictOfSeries(out)


@register(handles="data|flags", datamask="all", multivariate=True)
def genericProcess(
    data: DictOfSeries,
    field: str | Sequence[str],
    flags: Flags,
    func: GenericFunction,
    target: str | Sequence[str] = None,
    flag: float = UNFLAGGED,
    to_mask: float = UNFLAGGED,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
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
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str or list of str
        The variable(s) passed to func.
    flags : saqc.Flags
        Container to store flags of the data.
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

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
        The shape of the data may have changed relatively to the data input.
    flags : saqc.Flags
        The quality flags of data
        The flags shape may have changed relatively to the input flags.

    Note
    -----
    All the numpy functions are available within the generic expressions.

    Examples
    --------
    Compute the sum of the variables 'rainfall' and 'snowfall' and save the result to
    a (new) variable 'precipitation'

    >>> saqc.genericProcess(field=["rainfall", "snowfall"], target="precipitation'", func=lambda x, y: x + y)
    """

    fields = toSequence(field)
    targets = toSequence(target or []) or fields
    result = _execGeneric(
        Flags({f: flags[f] for f in fields}), data.loc[:, fields], func, to_mask=to_mask
    )

    # uodate data & flags
    for i, col in enumerate(targets):
        datacol = result.iloc[:, i]
        data[col] = datacol
        if col in flags and flags[col].index.equals(datacol.index) is False:
            raise ValueError(
                f"cannot assign function result to the existing variable {repr(col)} "
                "because of incompatible indices, please choose another 'target'"
            )

        flags[col] = pd.Series(flag, index=datacol.index)

    return data, flags


@register(handles="data|flags", datamask="all", multivariate=True)
def genericFlag(
    data: DictOfSeries,
    field: Union[str, Sequence[str]],
    flags: Flags,
    func: GenericFunction,
    target: Union[str, Sequence[str]] = None,
    flag: float = BAD,
    to_mask: float = UNFLAGGED,
    **kwargs,
) -> Tuple[DictOfSeries, Flags]:
    """
    Flag data with user defined functions.

    Formally, what the function does, is the following:
    Let X be a Callable, depending on fields f_1, f_2,...f_K, (X = X(f_1, f_2,...f_K))
    Than for every timestamp t_i in data[field]:
    data[field][t_i] is flagged if X(data[f_1][t_i], data[f_2][t_i], ..., data[f_K][t_i]) is True.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str or list of str
        The variable(s) passed to func.
    flags : saqc.Flags
        Container to store flags of the data.
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

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flags : saqc.Flags
        The quality flags of data
        Flags values may have changed relatively to the flags input.

    Note
    -----
    All the numpy functions are available within the generic expressions.

    Examples
    --------

    1. Flag the variable 'rainfall', if the sum of the variables 'temperature' and 'uncertainty' is below zero:

    >>> saqc.genericFlag(field=["temperature", "uncertainty"], target="rainfall", func= lambda x, y: temperature + uncertainty < 0

    2. Flag the variable 'temperature', where the variable 'fan' is flagged:

    >>> saqc.genericFlag(field="fan", target="temperature", func=lambda x: isflagged(x))

    3. The generic functions also support all pandas and numpy functions:

    >>> saqc.genericFlag(field="fan", target="temperature", func=lambda x: np.sqrt(x) < 7)
    """

    fields = toSequence(field)
    targets = toSequence(target or []) or fields
    result = _execGeneric(
        Flags({f: flags[f] for f in fields}),
        data.loc[:, fields],
        func,
        to_mask=to_mask,
    )

    if len(targets) != len(result.columns):
        raise ValueError(
            f"the generic function returned {len(result.columns)} field(s), but only {len(targets)} target(s) were given"
        )

    if not (result.dtypes == bool).all():
        raise TypeError(f"generic expression does not return a boolean array")

    # update flags & data
    for i, col in enumerate(targets):
        maskcol = result.iloc[:, i]
        if col in flags and not flags[col].index.equals(maskcol.index):
            raise ValueError(
                f"cannot assign function result to the existing variable {repr(col)} "
                "because of incompatible indices, please choose another 'target'"
            )
        flagcol = maskcol.replace({False: np.nan, True: flag})
        flags[col] = flagcol
        if col not in data:
            data[col] = pd.Series(np.nan, index=maskcol.index)

    return data, flags
