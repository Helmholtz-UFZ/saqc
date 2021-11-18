#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Sequence,
    Hashable,
)
from copy import deepcopy, copy as shallowcopy

import pandas as pd
import numpy as np

from dios import DictOfSeries, to_dios

from saqc.core.flags import initFlagsLike, Flags
from saqc.core.history import History
from saqc.core.register import FUNC_MAP, FunctionWrapper
from saqc.core.modules import FunctionsMixin
from saqc.core.translator import (
    Translator,
    FloatTranslator,
    SimpleTranslator,
    PositionalTranslator,
    DmpTranslator,
)
from saqc.lib.tools import toSequence
from saqc.lib.types import (
    ExternalFlag,
)

# the import is needed to trigger the registration
# of the built-in (test-)functions
import saqc.funcs  # noqa

# warnings
pd.set_option("mode.chained_assignment", "warn")
np.seterr(invalid="ignore")


TRANSLATION_SCHEMES = {
    "float": FloatTranslator,
    "simple": SimpleTranslator,
    "dmp": DmpTranslator,
    "positional": PositionalTranslator,
}


class SaQC(FunctionsMixin):
    _attributes = {
        "_data",
        "_flags",
        "_translator",
        "_attrs",
        "_called",
    }

    def __init__(
        self,
        data=None,
        flags=None,
        scheme: str | Translator = "float",
        copy: bool = True,
    ):
        self._data = self._initData(data, copy)
        self._flags = self._initFlags(flags, copy)
        self._translator = self._initTranslator(scheme)  # scheme or FloatTranslator()
        self._called = []
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
        out = SaQC(data=DictOfSeries(), flags=Flags(), scheme=self._translator)
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
    def dataRaw(self) -> DictOfSeries:
        return self._data

    @property
    def flagsRaw(self) -> Flags:
        return self._flags

    @property
    def data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._data.to_df()
        data.attrs = self._attrs.copy()
        return data

    @property
    def flags(self) -> pd.DataFrame:
        data: pd.DataFrame = self._translator.backward(self._flags, attrs=self._attrs)
        data.attrs = self._attrs.copy()
        return data

    @property
    def result(self) -> SaQCResult:
        return SaQCResult(self._data, self._flags, self._attrs, self._translator)

    def _wrap(self, func: FunctionWrapper):
        """Enrich a function by special saqc-functionality.

        For each saqc function this realize
            - regex's in field,
            - use default of translator for ``to_mask`` if not specified by user,
            - translation of ``flag`` and
            - working inplace.
        Therefore it adds the following keywords to each saqc function:
        ``regex`` and ``inplace``.

        The returned function returns a Saqc object.
        """

        def inner(
            field: str | Sequence[str],
            *args,
            regex: bool = False,
            flag: ExternalFlag = None,
            **kwargs,
        ) -> SaQC:

            kwargs.setdefault("to_mask", self._translator.TO_MASK)

            # translation
            if flag is not None:
                kwargs["flag"] = self._translator(flag)

            # expand regular expressions
            if regex:
                fmask = self._data.columns.str.match(field)
                fields = self._data.columns[fmask].tolist()
            else:
                fields = toSequence(field)

            # we wrap field again to generalize the
            # down stream loop work as expected
            if func._multivariate:
                fields = [fields]

            out = self

            for f in fields:
                out = out._callFunction(func, field=f, *args, **kwargs)
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
        self._called += [(field, (function, args, kwargs))]
        self._validate(reason=f"Call to {repr(function.__name__)}")

        return self._construct(
            _data=self._data, _flags=self._flags, _called=self._called
        )

    def __getattr__(self, key):
        """
        All failing attribute accesses are redirected to
        __getattr__. We use this mechanism to make the
        registered functions as `SaQC`-methods without
        actually implementing them.
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

    def _initTranslator(self, scheme: str | Translator) -> Translator:
        if isinstance(scheme, str) and scheme in TRANSLATION_SCHEMES:
            return TRANSLATION_SCHEMES[scheme]()
        if isinstance(scheme, Translator):
            return scheme
        raise TypeError(
            f"expected one of the following translation schemes '{TRANSLATION_SCHEMES.keys()} "
            f"or an initialized Translator object, got '{scheme}'"
        )

    def _initData(self, data, copy: bool) -> DictOfSeries:
        if data is None:
            data = DictOfSeries()
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if not isinstance(data, (DictOfSeries, pd.DataFrame)):
            raise TypeError(
                "'data' must be of type pandas.Series, "
                "pandas.DataFrame or dios.DictOfSeries"
            )
        if isinstance(data, pd.DataFrame):
            for idx in [data.index, data.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise TypeError("'data' should not have MultiIndex")

        data = to_dios(data)  # noop for DictOfSeries

        for c in data.columns:
            if not isinstance(c, str):
                raise TypeError("columns labels must be of type string")
        if copy:
            return data.copy()
        return data

    def _initFlags(self, flags, copy: bool) -> Flags:
        if isinstance(flags, pd.DataFrame):
            for idx in [flags.index, flags.columns]:
                if isinstance(idx, pd.MultiIndex):
                    raise TypeError("'flags' should not have MultiIndex")
        if flags is None:
            flags = initFlagsLike(self._data)
        elif isinstance(flags, (pd.DataFrame, DictOfSeries, Flags)):
            if not isinstance(flags, Flags):
                flags = Flags(flags)
            if copy:
                flags = flags.copy()
            for c in self._data.columns:
                if c not in flags.columns:
                    flags.history = History(self._data[c].index)
                else:
                    if not flags[c].index.equals(self._data[c].index):
                        raise ValueError(
                            f"index of 'flags' does not equal "
                            f"index of 'data' for column {c} "
                        )
        else:
            raise TypeError(
                "'flags' must be of type pandas.DataFrame, dios.DictOfSeries or Flags"
            )
        return flags


class SaQCResult:
    def __init__(
        self, data: DictOfSeries, flags: Flags, attrs: dict, translator: Translator
    ):
        assert isinstance(data, DictOfSeries)
        assert isinstance(flags, Flags)
        assert isinstance(attrs, dict)
        assert isinstance(translator, Translator)
        self._data = data.copy()
        self._flags = flags.copy()
        self._attrs = attrs.copy()
        self._translator = translator
        self._validate()

        try:
            self._translator.backward(self._flags, attrs=self._attrs)
        except Exception as e:
            raise RuntimeError("Translation of flags failed") from e

    def _validate(self):
        if not self._data.columns.equals(self._flags.columns):
            raise AssertionError(
                "Consistency broken. data and flags have not the same columns"
            )

    @property
    def data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._data.copy().to_df()
        data.attrs = self._attrs.copy()
        return data

    @property
    def flags(self) -> pd.DataFrame:
        data: pd.DataFrame = self._translator.backward(self._flags, attrs=self._attrs)
        data.attrs = self._attrs.copy()
        return data

    @property
    def dataRaw(self) -> DictOfSeries:
        return self._data

    @property
    def flagsRaw(self) -> Flags:
        return self._flags

    @property
    def columns(self) -> DictOfSeries():
        self._validate()
        return self._data.columns

    def __getitem__(self, key):
        self._validate()
        if key not in self.columns:
            raise KeyError(key)
        data_series = self._data[key].copy()
        # slice flags to one column
        flags = Flags({key: self._flags._data[key]}, copy=True)

        df = self._translator.backward(flags, attrs=self._attrs)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=0, axis=1)

        if len(df.columns) == 1:
            df.columns = ["flags"]

        df.insert(0, column="data", value=data_series)
        df.columns.name = None
        df.index.name = None
        return df

    def __repr__(self):
        return f"SaQCResult\nColumns: {self.columns.to_list()}"
