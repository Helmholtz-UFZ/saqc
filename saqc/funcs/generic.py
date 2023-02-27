#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from saqc import BAD, FILTER_ALL
from saqc.core import DictOfSeries, Flags, History, register
from saqc.core.register import _maskData
from saqc.lib.tools import isAllBoolean, isflagged, toSequence
from saqc.lib.types import GenericFunction
from saqc.parsing.environ import ENVIRONMENT

if TYPE_CHECKING:
    from saqc import SaQC


def _flagSelect(field, flags, label=None):
    if label is None:
        return flags[field]

    h_meta = flags.history[field].meta
    trg_col = None

    for idx, item in enumerate(h_meta):
        kwargs = item.get("kwargs")
        if kwargs is None or "label" not in kwargs:
            continue
        if kwargs["label"] == label:
            trg_col = idx

    if trg_col is None:
        raise KeyError(f"no such label {label} for field {field}")

    out = flags.history[field].hist[trg_col].astype(float)
    return out.fillna(-np.inf)


def _prepare(
    data: DictOfSeries, flags: Flags, columns: Sequence[str], dfilter: float
) -> Tuple[DictOfSeries, Flags]:
    fchunk = Flags({f: flags[f] for f in columns})
    for f in fchunk.columns:
        fchunk.history[f] = flags.history[f]
    dchunk, _ = _maskData(
        data=data[columns].copy(), flags=fchunk, columns=columns, thresh=dfilter
    )
    return dchunk, fchunk.copy()


def _execGeneric(
    flags: Flags,
    data: pd.DataFrame | pd.Series | DictOfSeries,
    func: GenericFunction,
    dfilter: float = FILTER_ALL,
) -> DictOfSeries:
    globs = {
        "isflagged": lambda data, label=None: isflagged(
            _flagSelect(data.name, flags, label), thresh=dfilter
        ),
        **ENVIRONMENT,
    }

    func.__globals__.update(globs)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # set series.name, because `isflagged` relies on it
    cols = []
    for c in data.columns:
        data[c].name = c
        cols.append(data[c])
    return func(*cols)


def _castResult(obj) -> DictOfSeries:
    # Note: the actual keys aka. column names
    # we use here to create a DictOfSeries
    # are never used, and only exists temporary.

    if isinstance(obj, pd.Series):
        return DictOfSeries({"0": obj})
    if pd.api.types.is_dict_like(obj):
        # includes pd.Series and
        # everything with keys and __getitem__
        return DictOfSeries(obj)
    if pd.api.types.is_list_like(obj):
        # includes pd.Series and dict
        return DictOfSeries({str(i): val for i, val in enumerate(obj)})

    if pd.api.types.is_scalar(obj):
        raise TypeError(
            "generic function should return a sequence object, "
            f"got '{type(obj)}' instead"
        )
    raise TypeError(f"unprocessable result type {type(obj)}.")


class GenericMixin:
    @register(
        mask=[],
        demask=[],
        squeeze=[],
        multivariate=True,
        handles_target=True,
    )
    def processGeneric(
        self: "SaQC",
        field: str | Sequence[str],
        func: GenericFunction,
        target: str | Sequence[str] | None = None,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> "SaQC":
        """
        Generate/process data with user defined functions.

        Call the given ``func`` on the variables given in ``field``.

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

        flag: float, default ``np.nan``
            The quality flag to set. The default ``np.nan`` states the general idea, that
            ``processGeneric`` generates 'new' data without any flags.

        dfilter: float, default ``FILTER_ALL``
            Threshold flag. Flag values greater than ``dfilter`` indicate that the associated
            data value is inappropiate for further usage.

        Returns
        -------
        saqc.SaQC

        Note
        -----
        All the numpy functions are available within the generic expressions.

        Examples
        --------
        Compute the sum of the variables 'rainfall' and 'snowfall' and save the result to
        a (new) variable 'precipitation'

        >>> from saqc import SaQC
        >>> qc = SaQC(pd.DataFrame({'rainfall':[1], 'snowfall':[2]}, index=pd.DatetimeIndex([0])))
        >>> qc = qc.processGeneric(field=["rainfall", "snowfall"], target="precipitation", func=lambda x, y: x + y)
        >>> qc.data.to_pandas()
                    rainfall  snowfall  precipitation
        1970-01-01         1         2              3
        """

        fields = toSequence(field)
        targets = fields if target is None else toSequence(target)

        dchunk, fchunk = _prepare(self._data, self._flags, fields, dfilter)
        result = _execGeneric(fchunk, dchunk, func, dfilter=dfilter)
        result = _castResult(result)

        meta = {
            "func": "procGeneric",
            "args": (field, target),
            "kwargs": {
                "func": func.__name__,
                "dfilter": dfilter,
                **kwargs,
            },
        }

        # update data & flags
        for i, col in enumerate(targets):
            datacol = result[result.columns[i]]
            self._data[col] = datacol

            if col not in self._flags:
                self._flags.history[col] = History(datacol.index)

            if not self._flags[col].index.equals(datacol.index):
                raise ValueError(
                    f"cannot assign function result to the existing variable {repr(col)} "
                    "because of incompatible indices, please choose another 'target'"
                )

            self._flags.history[col].append(
                pd.Series(np.nan, index=datacol.index), meta
            )

        return self

    @register(
        mask=[],
        demask=[],
        squeeze=[],
        multivariate=True,
        handles_target=True,
    )
    def flagGeneric(
        self: "SaQC",
        field: str | Sequence[str],
        func: GenericFunction,
        target: str | Sequence[str] | None = None,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> "SaQC":
        """
        Flag data based on a given function.

        Evaluate ``func`` on all variables given in ``field``.

        Parameters
        ----------
        field : str or list of str
            The variable(s) passed to func.

        func : callable
            Function to call. The function needs to accept the same number of arguments
            (of type pandas.Series) as variables given in ``field`` and return an
            iterable of array-like objects of data type ``bool`` with the same length as
            ``target``.

        target: str or list of str
            The variable(s) to write the result of ``func`` to. If not given, the variable(s)
            specified in ``field`` will be overwritten. Non-existing ``target``s  will be created
            as all ``NaN`` timeseries.

        flag: float, default ``BAD``
            Quality flag to set.

        dfilter: float, default ``FILTER_ALL``
            Threshold flag. Flag values greater than ``dfilter`` indicate that the associated
            data value is inappropiate for further usage.

        Returns
        -------
        saqc.SaQC

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

        dchunk, fchunk = _prepare(self._data, self._flags, fields, dfilter)
        result = _execGeneric(fchunk, dchunk, func, dfilter=dfilter)
        result = _castResult(result)

        if len(targets) != len(result.columns):
            raise ValueError(
                f"the generic function returned {len(result.columns)} field(s), "
                f"but {len(targets)} target(s) were given"
            )

        if not result.empty and not isAllBoolean(result):
            raise TypeError(f"generic expression does not return a boolean array")

        meta = {
            "func": "flagGeneric",
            "args": (field, target),
            "kwargs": {
                "func": func.__name__,
                "flag": flag,
                "dfilter": dfilter,
                **kwargs,
            },
        }

        # update flags & data
        for i, col in enumerate(targets):
            maskcol = result[result.columns[i]]

            # make sure the column exists
            if col not in self._flags:
                self._flags.history[col] = History(maskcol.index)

            # dummy column to ensure consistency between flags and data
            if col not in self._data:
                self._data[col] = pd.Series(np.nan, index=maskcol.index, dtype=float)

            # Note: big speedup for series, because replace works
            # with a loop and setting with mask is vectorized.
            # old code:
            # >>> flagcol = maskcol.replace({False: np.nan, True: flag}).astype(float)
            flagcol = pd.Series(np.nan, index=maskcol.index, dtype=float)
            flagcol[maskcol] = flag

            # we need equal indices to work on
            if not self._flags[col].index.equals(maskcol.index):
                raise ValueError(
                    f"cannot assign function result to the existing variable {repr(col)} "
                    "because of incompatible indices, please choose another 'target'"
                )

            self._flags.history[col].append(flagcol, meta)

        return self
