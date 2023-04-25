#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from typing_extensions import ParamSpec

from saqc import FILTER_ALL, FILTER_NONE
from saqc.core import DictOfSeries, Flags, History
from saqc.core.translation.basescheme import TranslationScheme
from saqc.lib.docs import ParamDict, docurator
from saqc.lib.tools import isflagged, squeezeSequence, toSequence
from saqc.lib.types import ExternalFlag, OptionalNone

if TYPE_CHECKING:
    from saqc import SaQC

__all__ = [
    "register",
    "processing",
    "flagging",
]

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, Callable] = {}

_is_list_like = pd.api.types.is_list_like

T = TypeVar("T")
P = ParamSpec("P")


def _checkDecoratorKeywords(
    func_signature, func_name, mask, demask, squeeze, handles_target
):
    params = func_signature.parameters.keys()
    if "target" in params and not handles_target:
        raise TypeError(
            "functions defining a parameter named 'target' "
            "need to decorated with 'handles_target=True'"
        )
    for dec_arg, name in zip([mask, demask, squeeze], ["mask", "demask", "squeeze"]):
        typeerr = TypeError(
            f"type of decorator argument '{name}' must "
            f"be a list of strings, not {repr(type(dec_arg))}"
        )
        if not isinstance(dec_arg, list):
            raise typeerr
        for elem in dec_arg:
            if not isinstance(elem, str):
                raise typeerr
            if elem not in params:
                raise ValueError(
                    f"passed value {repr(elem)} in {repr(name)} is not an "
                    f"parameter in decorated function {repr(func_name)}"
                )


def _argnamesToColumns(names: list, values: dict):
    clist = []
    for name in names:
        value = values.get(name)  # eg. the value behind 'field'

        # NOTE: do not change order of the tests
        if value is None:
            pass
        elif isinstance(value, str):
            clist.append(value)
        # we ignore DataFrame, Series, DictOfSeries
        # and high order types alike
        elif hasattr(value, "columns"):
            pass
        elif _is_list_like(value) and all([isinstance(e, str) for e in value]):
            clist += value
    return pd.Index(clist)


def _warn(missing, source):
    if len(missing) == 0:
        return
    action = source + "ed"
    obj = "flags" if source == "squeeze" else "data"
    warnings.warn(
        f"Column(s) {repr(missing)} cannot not be {action} "
        f"because they are not present in {obj}. ",
        RuntimeWarning,
    )


def _getDfilter(
    func_signature: inspect.Signature,
    translation_scheme: TranslationScheme,
    kwargs: Dict[str, Any],
) -> float:
    """
    Find a default value for dfilter, either from the choosen translation scheme
    or a possibly defined method default value. Translate, if necessary.
    """
    dfilter = kwargs.get("dfilter")
    if dfilter is None or isinstance(dfilter, OptionalNone):
        # let's see, if the function has a default value
        default = func_signature.parameters.get("dfilter")
        if default is None or default.default == inspect.Signature.empty:
            default = FILTER_ALL
        else:
            default = default.default
        dfilter = max(translation_scheme.DFILTER_DEFAULT, default)
    else:
        # try to translate dfilter
        if dfilter not in {FILTER_ALL, FILTER_NONE, translation_scheme.DFILTER_DEFAULT}:
            dfilter = translation_scheme(dfilter)
    return float(dfilter)


def _squeezeFlags(old_flags, new_flags: Flags, columns: pd.Index, meta) -> Flags:
    """
    Generate flags from the temporary result-flags and the original flags.

    Parameters
    ----------
    flags : Flags
        The flags-frame, which is the result from a saqc-function

    Returns
    -------
    Flags
    """
    out = old_flags.copy()  # the old flags

    for col in columns.union(
        new_flags.columns.difference(old_flags.columns)
    ):  # account for newly added columns
        if col not in out:  # ensure existence
            out.history[col] = History(index=new_flags.history[col].index)

        old_history = out.history[col]
        new_history = new_flags.history[col]

        # We only want to add new columns, that were appended during the last
        # function call. If no such columns exist, we end up with an empty
        # new_history.
        start = len(old_history.columns)
        squeezed = new_history.squeeze(raw=True, start=start)
        out.history[col] = out.history[col].append(squeezed, meta=meta)

    return out


def _maskData(
    data: DictOfSeries, flags: Flags, columns: Sequence[str], thresh: float
) -> Tuple[DictOfSeries, DictOfSeries]:
    """
    Mask data with Nans, if the flags are worse than a threshold.
        - mask only passed `columns` (preselected by `datamask`-kw from decorator)

    Returns
    -------
    masked : DictOfSeries
        masked data, same dim as original
    mask : DictOfSeries
        dios holding iloc-data-pairs for every column in `data`
    """
    mask = DictOfSeries()

    # we use numpy here because it is faster
    for c in columns:
        col_mask = isflagged(flags[c].to_numpy(), thresh)

        if col_mask.any():
            col_data = data[c].to_numpy(dtype=np.float64, copy=True)
            mask[c] = pd.Series(col_data[col_mask], index=np.where(col_mask)[0])
            col_data[col_mask] = np.nan
            data[c] = pd.Series(col_data, index=data[c].index)

    return data, mask


def _unmaskData(
    data: DictOfSeries, mask: DictOfSeries, columns: pd.Index | None = None
) -> DictOfSeries:
    """
    Restore the masked data.

    Notes
    -----
    - Even if this returns data, it works inplace !
    - `mask` is not a boolean mask, instead it holds the original values.
      The index of mask is numeric and represent the integer location
      in the original data.
    """
    if columns is None:
        columns = data.columns  # field was in old, is in mask and is in new
    columns = mask.columns.intersection(columns)

    for c in columns:
        # ignore
        if data[c].empty or mask[c].empty:
            continue

        # get the positions of values to unmask
        candidates = mask[c]
        # if the mask was removed during the function call, don't replace
        unmask = candidates[data[c].iloc[candidates.index].isna().to_numpy()]
        if unmask.empty:
            continue
        data[c].iloc[unmask.index] = unmask

    return data


def _expandField(regex, columns, field) -> List[str]:
    """
    Expand regular expressions to concrete field names.
    """
    if regex:
        fmask = columns.str.match(field)
        return columns[fmask].tolist()
    return toSequence(field)


def _homogenizeFieldsTargets(
    multivariate,
    handles_target,
    fields,
    targets,
):
    """
    Ensure, that fields and flags are of identical length.

    Note
    ----
    We have four cases to consider:
    1. multivariate=False & handles_target=False
    2. multivariate=False & handles_target=True
    3. multivariate=True  & handles_target=False
    4. multivariate=True  & handles_target=True
    """

    if not (multivariate and handles_target):
        if len(fields) != len(targets):
            raise ValueError("expected the same number of 'field' and 'target' values")

    if multivariate:
        fields, targets = [fields], [targets]

    return fields, targets


def register(
    mask: list[str],
    demask: list[str],
    squeeze: list[str],
    multivariate: bool = False,
    handles_target: bool = False,
    docstring: dict[str, ParamDict] | None = None,
):
    """
    Generalized decorator for any saqc functions.

    Before the call of the decorated function:
    - data gets masked by flags according to `dfilter`

    After the call of the decorated function:
    - data gets demasked (original data is written back)
    - flags gets squeezed (only one history column append per call)

    Parameters
    ----------
    mask : list of string
        List of parameters of the decorated function, that specify column(s) in
        ``SaQC._data``, that are read by the function and therefore should be masked.

        The masking takes place before the call of the decorated function and
        temporary sets data to `NaN` at flagged locations. It is undone by ``demask``.
        The threshold of which data is considered to be flagged can be controlled
        via ``dfilter``, a parameter each function takes.

    demask : list of string
        List of parameters of the decorated function, that specify column(s) in
        ``SaQC._data``, that were masked (see ``mask``) and needs unmasking after the call.

        The unmasking replaces all remaining(!) ``NaN`` inserted in the masking process by
        its their original values.

    squeeze : list of string
        List of parameters of the decorated function, that specify flag column(s),
        that are written by the function.

        The squeezing combines multiple columns in the history of flags to one
        single column. This is because, multiple writes to flags, (eg. using
        ``flags[:,'a'] = 255`` twice) will result in multiple history columns,
        but should considered as a single column, because only one function call
        happened.

    multivariate : bool, default False
        If ``True``, the decorated function, processes multiple data or flag
        columns at once. Therefore the decorated function must handle a list
        of columns in the parameter ``field``.

        If ``False``, the decorated function must define ``field: str``

    handles_target : bool, default False
        If ``True``, the decorated function, handles the target parameter by
        itself. Mandatory for multivariate functions.

    docstring : dict, default None
        Allows to modify the default docstring description of the parameters ``field``,
        ``target``, ``dfilter`` and ``flag``

    """

    def outer(func: Callable[P, SaQC]) -> Callable[P, SaQC]:
        func_signature = inspect.signature(func)
        _checkDecoratorKeywords(
            func_signature, func.__name__, mask, demask, squeeze, handles_target
        )
        func = docurator(func, docstring)

        @functools.wraps(func)
        def inner(
            saqc,
            field,
            *args,
            regex: bool = False,
            flag: ExternalFlag | OptionalNone = OptionalNone(),
            **kwargs,
        ) -> "SaQC":
            # args -> kwargs
            paramnames = tuple(func_signature.parameters.keys())[
                2:
            ]  # skip (self, field)

            # check for duplicated arguments
            args_map = dict(zip(paramnames, args))
            intersection = set(args_map).intersection(set(kwargs))
            if intersection:
                raise TypeError(
                    f"SaQC.{func.__name__}() got multiple values for argument '{intersection.pop()}'"
                )
            kwargs = {**args_map, **kwargs}
            kwargs["dfilter"] = _getDfilter(func_signature, saqc._scheme, kwargs)

            # translate flag
            if not isinstance(flag, OptionalNone):
                # translation schemes might want to use a flag
                # `None` so we introduce a special class here
                kwargs["flag"] = saqc._scheme(flag)

            fields = _expandField(regex, saqc._data.columns, field)
            targets = toSequence(kwargs.pop("target", fields))

            fields, targets = _homogenizeFieldsTargets(
                multivariate, handles_target, fields, targets
            )

            out = saqc.copy(deep=True)

            # initialize target fields
            if not handles_target:
                # initialize all target variables
                for src, trg in zip(fields, targets):
                    if src != trg:
                        out = out.copyField(field=src, target=trg, overwrite=True)

            for src, trg in zip(fields, targets):
                kwargs = {**kwargs, "field": src, "target": trg}
                if not handles_target:
                    kwargs["field"] = kwargs.pop("target")

                # find columns that need masking
                # func_signature = func_signature.bind(field=field)
                columns = _argnamesToColumns(mask, kwargs)
                _warn(columns.difference(out._data.columns).to_list(), source="mask")
                columns = columns.intersection(out._data.columns)

                out._data, stored_data = _maskData(
                    data=out._data,
                    flags=out._flags,
                    columns=columns,
                    thresh=kwargs["dfilter"],
                )

                # always pass a list to multivariate functions and
                # unpack single element lists for univariate functions
                if not multivariate:
                    kwargs["field"] = squeezeSequence(kwargs["field"])

                old_flags = out._flags.copy()

                out = func(out, **kwargs)

                # find columns that need squeezing
                columns = _argnamesToColumns(squeeze, kwargs)
                _warn(
                    columns.difference(out._flags.columns).to_list(), source="squeeze"
                )
                columns = columns.intersection(out._flags.columns)

                # if the function did not want to set any flags at all,
                # we assume a processing function that altered the flags
                # in an unpredictable manner or do nothing with the flags.
                # in either case we take the returned flags as the new truth.
                if not columns.empty:
                    meta = {
                        "func": func.__name__,
                        "args": args,
                        "kwargs": kwargs,
                    }
                    out._flags = _squeezeFlags(old_flags, out._flags, columns, meta)

                # find columns that need demasking
                columns = _argnamesToColumns(demask, kwargs)
                _warn(columns.difference(out._data.columns).to_list(), source="demask")
                columns = columns.intersection(out._data.columns)

                out._data = _unmaskData(out._data, stored_data, columns=columns)
                out._validate(reason=f"call to {repr(func.__name__)}")

            return out

        FUNC_MAP[func.__name__] = inner
        return inner

    return outer


def flagging(**kwargs):
    """
    Default decorator for univariate flagging functions.

    Before the call of the decorated function:
    - `data[field]` gets masked by `flags[field]` according to `dfilter`
    After the call of the decorated function:
    - `data[field]` gets demasked (original data is written back)
    - `flags[field]` gets squeezed (only one history column append per call) if needed

    Notes
    -----
    For full control over masking, demasking and squeezing or to implement
    a multivariate function (multiple in- or outputs) use the `@register` decorator.

    See Also
    --------
        resister: generalization of of this function
    """
    if kwargs:
        raise ValueError("use '@register' to pass keywords")
    return register(mask=["field"], demask=["field"], squeeze=["field"])


def processing(**kwargs):
    """
    Default decorator for univariate processing functions.

    - no masking of data
    - no demasking of data
    - no squeezing of flags

    Notes
    -----
    For full control over masking, demasking and squeezing or to implement
    a multivariate function (multiple in- or outputs) use the `@register` decorator.

    See Also
    --------
        resister: generalization of of this function
    """
    if kwargs:
        raise ValueError("use '@register' to pass keywords")
    return register(mask=[], demask=[], squeeze=[])
