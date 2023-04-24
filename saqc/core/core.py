#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from copy import copy as shallowcopy
from copy import deepcopy
from typing import Any, Hashable, MutableMapping

import numpy as np
import pandas as pd

from saqc.core.flags import Flags, _HistAccess, initFlagsLike
from saqc.core.frame import DictOfSeries
from saqc.core.history import History
from saqc.core.register import FUNC_MAP
from saqc.core.translation import (
    DmpScheme,
    FloatScheme,
    PositionalScheme,
    SimpleScheme,
    TranslationScheme,
)
from saqc.funcs import FunctionsMixin

# warnings
pd.set_option("mode.chained_assignment", "warn")
pd.options.mode.copy_on_write = False
np.seterr(invalid="ignore")


TRANSLATION_SCHEMES = {
    "float": FloatScheme,
    "simple": SimpleScheme,
    "dmp": DmpScheme,
    "positional": PositionalScheme,
}


class SaQC(FunctionsMixin):
    _attributes = {
        "_data",
        "_flags",
        "_scheme",
        "_attrs",
    }

    def __init__(
        self,
        data=None,
        flags=None,
        scheme: str | TranslationScheme = "float",
    ):
        self._data: DictOfSeries = self._initData(data)
        self._flags: Flags = self._initFlags(flags)
        self._scheme: TranslationScheme = self._initTranslationScheme(scheme)
        self._attrs: dict = {}
        self._validate(reason="init")

    def _construct(self, **attributes) -> "SaQC":
        """
        Construct a new `SaQC`-Object from `self` and optionally inject
        attributes with any chechking and overhead.

        Parameters
        ----------
        **attributes: any of the `SaQC` data attributes with name and value

        Note
        ----
        For internal usage only! Setting values through `injectables` has
        the potential to mess up certain invariants of the constructed object.
        """
        out = SaQC(data=DictOfSeries(), flags=Flags(), scheme=self._scheme)
        out.attrs = self._attrs
        for k, v in attributes.items():
            if k not in self._attributes:
                raise AttributeError(f"SaQC has no attribute {repr(k)}")
            setattr(out, k, v)
        return out

    def _validate(self, reason=None):
        if not self._data.columns.equals(self._flags.columns):
            msg = "Consistency broken. data and flags have not the same columns."
            if reason:
                msg += f" This was most likely caused by: {reason}"
            raise RuntimeError(msg)

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: dict[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def data(self) -> MutableMapping:
        data = self._data
        data.attrs = self._attrs.copy()
        return data

    @property
    def flags(self) -> MutableMapping:
        flags = self._scheme.toExternal(self._flags, attrs=self._attrs)
        flags.attrs = self._attrs.copy()
        return flags

    @property
    def _history(self) -> _HistAccess:
        return self._flags.history

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to __getattr__.
        We use this mechanism to make the registered functions appear
        as `SaQC`-methods without actually implementing them.
        """
        from functools import partial

        if key not in FUNC_MAP:
            raise AttributeError(f"SaQC has no attribute {repr(key)}")
        return partial(FUNC_MAP[key], self)

    def copy(self, deep=True):
        copyfunc = deepcopy if deep else shallowcopy
        new = self._construct()
        for attr in self._attributes:
            setattr(new, attr, copyfunc(getattr(self, attr)))
        return new

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memodict=None):
        return self.copy(deep=True)

    def _initTranslationScheme(
        self, scheme: str | TranslationScheme
    ) -> TranslationScheme:
        if isinstance(scheme, str) and scheme in TRANSLATION_SCHEMES:
            return TRANSLATION_SCHEMES[scheme]()
        if isinstance(scheme, TranslationScheme):
            return scheme
        raise TypeError(
            f"expected one of the following translation schemes '{TRANSLATION_SCHEMES.keys()} "
            f"or an initialized Translator object, got '{scheme}'"
        )

    def _initData(self, data) -> DictOfSeries:
        if data is None:
            return DictOfSeries()
        if isinstance(data, list):
            result = DictOfSeries()
            doubles = pd.Index([])
            for d in data:
                new = self._castData(d)
                doubles = doubles.union(result.columns.intersection(new.columns))
                result.update(new)
            if not doubles.empty:
                warnings.warn(
                    f"Column(s) {doubles.tolist()} was present multiple "
                    f"times in input data. Some data was overwritten. "
                    f"Avoid duplicate columns names over all inputs.",
                    stacklevel=2,
                )
            return result
        try:
            return self._castData(data)
        except ValueError as e:
            raise e from None
        except TypeError as e:
            raise TypeError(
                "'data' must be of type pandas.Series, "
                "pandas.DataFrame or saqc.DictOfSeries or "
                "a list of those or a dict with string keys "
                "and pandas.Series as values."
            ) from e

    def _castData(self, data) -> DictOfSeries:
        if isinstance(data, pd.Series):
            if not isinstance(data.name, str):
                raise ValueError(f"Cannot init from unnamed pd.Series")
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            for idx in [data.index, data.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise ValueError("'data' should not have MultiIndex")
        try:
            # This ensures that values are pd.Series
            return DictOfSeries(data)
        except Exception:
            raise TypeError(f"Cannot cast {type(data)} to DictOfSeries") from None

    def _initFlags(self, flags) -> Flags:
        if flags is None:
            return initFlagsLike(self._data)

        if isinstance(flags, list):
            result = Flags()
            for f in flags:
                f = self._castToFlags(f)
                for c in f.columns:
                    if c in result.columns:
                        warnings.warn(
                            f"Column {c} already exist. Data is overwritten. "
                            f"Avoid duplicate columns names over all inputs.",
                            stacklevel=2,
                        )
                        result.history[c] = f.history[c]
            flags = result

        elif isinstance(flags, (pd.DataFrame, DictOfSeries, Flags)):
            flags = self._castToFlags(flags)

        else:
            raise TypeError(
                "'flags' must be of type pandas.DataFrame, "
                "dios.DictOfSeries or saqc.Flags or "
                "a list of those."
            )

        # sanitize
        # - if column is missing flags but present in data, add it
        # - if column is present in both, the index must be equal
        for c in self._data.columns:
            if c not in flags.columns:
                flags.history[c] = History(self._data[c].index)
            else:
                if not flags[c].index.equals(self._data[c].index):
                    raise ValueError(
                        f"The flags index of column {c} does not equals "
                        f"the index of the same column in data."
                    )
        return flags

    def _castToFlags(self, flags):
        if isinstance(flags, pd.DataFrame):
            for idx in [flags.index, flags.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise TypeError("'flags' should not have MultiIndex")
        if not isinstance(flags, Flags):
            flags = Flags(flags)
        return flags
