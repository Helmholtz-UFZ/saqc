#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from saqc import BAD, FILTER_ALL
from saqc.core import DictOfSeries, Flags, register
from saqc.lib.tools import isAllBoolean, isflagged, isunflagged, toSequence
from saqc.lib.types import (
    GenericFunction,
    NewSaQCFields,
    SaQC,
    SaQCFields,
    ValidatePublicMembers,
)
from saqc.parsing.environ import ENVIRONMENT


def _flagSelect(field: str, flags: Flags, label: str | None = None) -> pd.Series:
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


def _execGeneric(
    flags: Flags,
    data: pd.DataFrame | pd.Series | DictOfSeries,
    func: GenericFunction,
    dfilter: float = FILTER_ALL,
) -> DictOfSeries | pd.DataFrame | pd.Series:
    globs = {
        "isflagged": lambda data, label=None: isflagged(
            _flagSelect(data.name, flags, label), thresh=dfilter
        ),
        **ENVIRONMENT,
    }

    # some function don't have a globals attribute (e.g. np.sum)
    if not hasattr(func, "__globals__"):
        func.__globals__ = {}
    func.__globals__.update(globs)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # set series.name, because `isflagged` relies on it
    cols = []
    for c in data.columns:
        data[c].name = c
        cols.append(data[c])
    return func(*cols)


def _inferBroadcast(obj, trg_shape) -> pd.DataFrame:
    # simple single value broadcasting
    if pd.api.types.is_scalar(obj):
        return np.full(trg_shape, obj)
    return obj


def _inferDF(obj, cols, index):
    # infer dataframe if result is numpy array of fitting shape
    if isinstance(obj, np.ndarray):
        lc = len(cols)
        li = len(index)
        if (obj.shape == (li, lc)) or (obj.shape == (li,)):
            return pd.DataFrame(obj, columns=cols, index=index)
    return obj


def _castResult(obj) -> DictOfSeries:
    # Note: the actual keys aka. column names
    # we use here to create a DictOfSeries
    # are never used and only exist temporarily.

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


class GenericMixin(ValidatePublicMembers):
    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field", "target"],
        multivariate=True,
        handles_target=True,
    )
    def processGeneric(
        self: SaQC,
        field: SaQCFields,  # SaQCFields,
        func: GenericFunction,
        target: SaQCFields | NewSaQCFields | None = None,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> SaQC:
        """
        Generate/process data with user defined functions.

        Call the given ``func`` on the variables given in ``field``.

        Parameters
        ----------
        func :
            Function to call on the variables given in ``field``. The return value will be written
            to ``target`` or ``field`` if the former is not given. This implies, that the function
            needs to accept the same number of arguments (of type pandas.Series) as variables given
            in ``field`` and should return an iterable of array-like objects with the same number
            of elements as given in ``target`` (or ``field`` if ``target`` is not specified).


        Note
        ----
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

        dchunk, fchunk = self._data[fields].copy(), self._flags[fields].copy()
        trg_idx = dchunk[dchunk.columns[0]].index
        result = _execGeneric(fchunk, dchunk, func, dfilter=dfilter)
        result = _inferBroadcast(result, (len(trg_idx), len(targets)))
        result = _inferDF(result, cols=targets, index=trg_idx)
        result = _castResult(result)

        # update data & flags
        for i, col in enumerate(targets):
            datacol = result[result.columns[i]]
            self._data[col] = datacol

            if col not in self._flags:
                self._flags[col] = pd.Series(np.nan, index=datacol.index)

            if not self._flags[col].index.equals(datacol.index):
                raise ValueError(
                    f"cannot assign function result to the existing variable {repr(col)} "
                    "because of incompatible indices, please choose another 'target'"
                )

            self._flags[:, col] = np.nan

        return self

    @register(
        mask=["field"],
        demask=["field"],
        squeeze=["field", "target"],
        multivariate=True,
        handles_target=True,
    )
    def flagGeneric(
        self: SaQC,
        field: SaQCFields,
        func: GenericFunction,
        target: SaQCFields | NewSaQCFields | None = None,
        flag: float = BAD,
        **kwargs,
    ) -> SaQC:
        """
        Flag data based on a given function.

        Evaluate ``func`` on all variables given in ``field``.

        Parameters
        ----------
        func :
            Function to call. The function needs to accept the same number of arguments
            (of type pandas.Series) as variables given in ``field`` and return an
            iterable of array-like objects of data type ``bool`` with the same length as
            ``target``.

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
        dfilter = kwargs.get("dfilter", BAD)

        dchunk, fchunk = self._data[fields].copy(), self._flags[fields].copy()
        trg_idx = dchunk[dchunk.columns[0]].index
        result = _execGeneric(fchunk, dchunk, func, dfilter=dfilter)
        result = _inferBroadcast(result, (len(trg_idx), len(targets)))
        result = _inferDF(result, cols=targets, index=trg_idx)
        result = _castResult(result)

        if len(result.columns) > 1 and len(targets) != len(result.columns):
            raise ValueError(
                f"the generic function returned {len(result.columns)} field(s), "
                f"but {len(targets)} target(s) were given"
            )

        if not result.empty and not isAllBoolean(result):
            raise TypeError("generic expression does not return a boolean array")

        # update flags & data
        for i, col in enumerate(targets):
            # broadcast one column results to all targets
            mask = result[result.columns[i if len(result.columns) > 1 else 0]]

            # make sure the column exists
            if col not in self._flags:
                self._flags[col] = pd.Series(np.nan, index=mask.index)

            # respect existing flags
            mask = isunflagged(self._flags[col], thresh=dfilter) & mask

            # dummy column to ensure consistency between flags and data
            if col not in self._data:
                self._data[col] = pd.Series(np.nan, index=mask.index, dtype=float)

            # we need equal indices to work on
            if not self._flags[col].index.equals(mask.index):
                raise ValueError(
                    f"cannot assign function result to the existing variable {repr(col)} "
                    "because of incompatible indices, please choose another 'target'"
                )

            self._flags[mask, col] = flag

        return self
