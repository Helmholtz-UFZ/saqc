#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from dios import DictOfSeries

from saqc.constants import BAD, UNFLAGGED, ENVIRONMENT, FILTER_ALL
from saqc.core.history import History
from saqc.lib.tools import toSequence
from saqc.lib.types import GenericFunction, PandasLike
from saqc.core.flags import Flags
from saqc.core.register import register, _isflagged, FunctionWrapper


def _prepare(
    data: DictOfSeries, flags: Flags, columns: Sequence[str], dfilter: float
) -> Tuple[DictOfSeries, Flags]:
    fchunk = Flags({f: flags[f] for f in columns})
    dchunk, _ = FunctionWrapper._maskData(
        data=data.loc[:, columns].copy(), flags=fchunk, columns=columns, thresh=dfilter
    )
    return dchunk, fchunk.copy()


def _execGeneric(
    flags: Flags,
    data: PandasLike,
    func: GenericFunction,
    dfilter: float = FILTER_ALL,
) -> DictOfSeries:

    globs = {
        "isflagged": lambda data: _isflagged(flags[data.name], thresh=dfilter),
        **ENVIRONMENT,
    }

    func.__globals__.update(globs)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    out = func(*[data[c] for c in data.columns])
    if pd.api.types.is_scalar(out):
        raise ValueError(
            "generic function should return a sequence object, "
            f"got '{type(out)}' instead"
        )

    return DictOfSeries(out)


@register(
    mask=[],
    demask=[],
    squeeze=[],
    multivariate=True,
    handles_target=True,
)
def processGeneric(
    data: DictOfSeries,
    field: str | Sequence[str],
    flags: Flags,
    func: GenericFunction,
    target: str | Sequence[str] = None,
    flag: float = UNFLAGGED,
    dfilter: float = FILTER_ALL,
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
        ``processGeneric`` generates 'new' data without direct relation to the potentially
        already present flags.
    dfilter: float, default ``FILTER_ALL``
        Threshold flag. Flag values greater than ``dfilter`` indicate that the associated
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

    .. testsetup::

       qc = saqc.SaQC(pd.DataFrame({'rainfall':[0], 'snowfall':[0], 'precipitation':[0]}, index=pd.DatetimeIndex([0])))


    >>> qc = qc.processGeneric(field=["rainfall", "snowfall"], target="precipitation'", func=lambda x, y: x + y)
    """

    fields = toSequence(field)
    targets = fields if target is None else toSequence(target)

    dchunk, fchunk = _prepare(data, flags, fields, dfilter)
    result = _execGeneric(fchunk, dchunk, func, dfilter=dfilter)

    meta = {
        "func": "procGeneric",
        "args": (),
        "kwargs": {
            **kwargs,
            "field": field,
            "func": func.__name__,
            "target": target,
            "flag": flag,
            "dfilter": dfilter,
        },
    }

    # update data & flags
    for i, col in enumerate(targets):

        datacol = result.iloc[:, i]
        data[col] = datacol

        if col not in flags:
            flags.history[col] = History(datacol.index)

        if not flags[col].index.equals(datacol.index):
            raise ValueError(
                f"cannot assign function result to the existing variable {repr(col)} "
                "because of incompatible indices, please choose another 'target'"
            )

        flags.history[col].append(pd.Series(flag, index=datacol.index), meta)

    return data, flags


@register(
    mask=[],
    demask=[],
    squeeze=[],
    multivariate=True,
    handles_target=True,
)
def flagGeneric(
    data: DictOfSeries,
    field: Union[str, Sequence[str]],
    flags: Flags,
    func: GenericFunction,
    target: Union[str, Sequence[str]] = None,
    flag: float = BAD,
    dfilter: float = FILTER_ALL,
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
        ``processGeneric`` generates 'new' data without direct relation to the potentially
        already present flags.
    dfilter: float, default ``FILTER_ALL``
        Threshold flag. Flag values greater than ``dfilter`` indicate that the associated
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

    .. testsetup:: exampleFlagGeneric

       qc = saqc.SaQC(pd.DataFrame({'temperature':[0], 'uncertainty':[0], 'rainfall':[0], 'fan':[0]}, index=pd.DatetimeIndex([0])))

    1. Flag the variable 'rainfall', if the sum of the variables 'temperature' and 'uncertainty' is below zero:

    .. testcode:: exampleFlagGeneric

       qc.flagGeneric(field=["temperature", "uncertainty"], target="rainfall", func= lambda x, y: x + y < 0)

    2. Flag the variable 'temperature', where the variable 'fan' is flagged:

    .. testcode:: exampleFlagGeneric

       qc.flagGeneric(field="fan", target="temperature", func=lambda x: isflagged(x))

    3. The generic functions also support all pandas and numpy functions:

    .. testcode:: exampleFlagGeneric

       qc = qc.flagGeneric(field="fan", target="temperature", func=lambda x: np.sqrt(x) < 7)
    """

    fields = toSequence(field)
    targets = fields if target is None else toSequence(target)

    dchunk, fchunk = _prepare(data, flags, fields, dfilter)
    result = _execGeneric(fchunk, dchunk, func, dfilter=dfilter)

    if len(targets) != len(result.columns):
        raise ValueError(
            f"the generic function returned {len(result.columns)} field(s), but only {len(targets)} target(s) were given"
        )

    if not result.empty and not (result.dtypes == bool).all():
        raise TypeError(f"generic expression does not return a boolean array")

    meta = {
        "func": "flagGeneric",
        "args": (),
        "kwargs": {
            **kwargs,
            "field": field,
            "func": func.__name__,
            "target": target,
            "flag": flag,
            "dfilter": dfilter,
        },
    }

    # update flags & data
    for i, col in enumerate(targets):

        maskcol = result.iloc[:, i]

        # make sure the column exists
        if col not in flags:
            flags.history[col] = History(maskcol.index)

        # dummy column to ensure consistency between flags and data
        if col not in data:
            data[col] = pd.Series(np.nan, index=maskcol.index)

        maskcol = maskcol & ~_isflagged(flags[col], dfilter)
        flagcol = maskcol.replace({False: np.nan, True: flag}).astype(float)

        # we need equal indices to work on
        if not flags[col].index.equals(maskcol.index):
            raise ValueError(
                f"cannot assign function result to the existing variable {repr(col)} "
                "because of incompatible indices, please choose another 'target'"
            )

        flags.history[col].append(flagcol, meta)

    return data, flags
