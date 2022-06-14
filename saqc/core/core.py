#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from copy import copy as shallowcopy
from copy import deepcopy
from typing import Any, Callable, Hashable, List, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

# the import is needed to trigger the registration
# of the built-in (test-)functions
import saqc.funcs  # noqa
from dios import DictOfSeries, to_dios
from saqc.core.flags import Flags, initFlagsLike
from saqc.core.history import History
from saqc.core.modules import FunctionsMixin
from saqc.core.register import FUNC_MAP, FunctionWrapper
from saqc.core.translation import (
    DmpScheme,
    FloatScheme,
    PositionalScheme,
    SimpleScheme,
    TranslationScheme,
)
from saqc.lib.tools import concatDios, toSequence
from saqc.lib.types import ExternalFlag, OptionalNone

# warnings
pd.set_option("mode.chained_assignment", "warn")
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
        self._data = self._initData(data)
        self._flags = self._initFlags(flags)
        self._scheme = self._initTranslationScheme(scheme)
        self._attrs = {}
        self._validate(reason="init")

    def _construct(self, **attributes) -> SaQC:
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
        flags = self._scheme.backward(self._flags, attrs=self._attrs, raw=True)
        flags.attrs = self._attrs.copy()
        return flags

    def _expandFields(
        self,
        regex: bool,
        field: str | Sequence[str],
        target: str | Sequence[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        check and expand `field` and `target`
        """

        # expand regular expressions
        if regex:
            fmask = self._data.columns.str.match(field)
            fields = self._data.columns[fmask].tolist()
        else:
            fields = toSequence(field)

        targets = fields if target is None else toSequence(target)

        return fields, targets

    def _wrap(self, func: FunctionWrapper):
        """
        prepare user function input:
          - expand fields and targets
          - translate user given ``flag`` values or set the default ``BAD``
          - translate user given ``dfilter`` values or set the scheme default
          - dependeing on the workflow: initialize ``target`` variables

        Here we add the following parameters to all registered functions, regardless
        of their repsective definition:
          - ``regex``
          - ``target``

        """

        def inner(
            field: str | Sequence[str],
            *args,
            target: str | Sequence[str] = None,
            regex: bool = False,
            flag: ExternalFlag | OptionalNone = OptionalNone(),
            **kwargs,
        ) -> SaQC:

            if "dfilter" not in kwargs:
                # let's see, if the function has an default value
                default = func.func_signature.parameters.get("dfilter")
                if default:
                    default = default.default
                kwargs["dfilter"] = default or self._scheme.DFILTER_DEFAULT

            if not isinstance(flag, OptionalNone):
                # translation schemes might want to use a flag
                # `None` so we introduce a special class here
                kwargs["flag"] = self._scheme(flag)

            fields, targets = self._expandFields(
                regex=regex, field=field, target=target
            )
            out = self.copy(deep=True)

            if not func.handles_target:
                if len(fields) != len(targets):
                    raise ValueError(
                        "expected the same number of 'field' and 'target' values"
                    )

                # initialize all target variables
                for src, trg in zip(fields, targets):
                    if src != trg:
                        out = out._callFunction(
                            FUNC_MAP["copyField"],
                            field=src,
                            target=trg,
                            overwrite=True,
                        )

            if func.multivariate:
                # pass all fields and targets
                out = out._callFunction(
                    func,
                    field=fields,
                    target=targets,
                    *args,
                    **kwargs,
                )
            else:
                # call the function on target
                for src, trg in zip(fields, targets):
                    fkwargs = {**kwargs, "field": src, "target": trg}
                    if not func.handles_target:
                        fkwargs["field"] = fkwargs.pop("target")
                    out = out._callFunction(func, *args, **fkwargs)

            return out

        return inner

    def _callFunction(
        self,
        function: Callable,
        field: str | Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> SaQC:

        res = function(data=self._data, flags=self._flags, field=field, *args, **kwargs)

        # keep consistence: if we modify data and flags inplace in a function,
        # but data is the original and flags is a copy (as currently implemented),
        # data and flags of the original saqc obj may change inconsistently.
        self._data, self._flags = res
        self._validate(reason=f"call to {repr(function.__name__)}")

        return self._construct(_data=self._data, _flags=self._flags)

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to __getattr__.
        We use this mechanism to make the registered functions appear
        as `SaQC`-methods without actually implementing them.
        """
        if key not in FUNC_MAP:
            raise AttributeError(f"SaQC has no attribute {repr(key)}")
        return self._wrap(FUNC_MAP[key])

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
            results = []
            for d in data:
                results.append(self._castToDios(d))
            return concatDios(results, warn=True, stacklevel=3)

        if isinstance(data, (DictOfSeries, pd.DataFrame, pd.Series)):
            return self._castToDios(data)

        raise TypeError(
            "'data' must be of type pandas.Series, "
            "pandas.DataFrame or dios.DictOfSeries or"
            "a list of those."
        )

    def _castToDios(self, data):
        if isinstance(data, pd.Series):
            if not isinstance(data.name, str):
                raise ValueError(f"Cannot init from unnamed pd.Series")
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            for idx in [data.index, data.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise TypeError("'data' should not have MultiIndex")
        data = to_dios(data)  # noop for DictOfSeries
        for c in data.columns:
            if not isinstance(c, str):
                raise TypeError("columns labels must be of type string")
        return data

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
