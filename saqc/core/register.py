#!/usr/bin/env python
from __future__ import annotations
from typing import Dict, List, Union, Tuple, Callable, Sequence
from typing_extensions import Literal
from functools import wraps
import dataclasses
from dios.dios.dios import DictOfSeries
import numpy as np
import pandas as pd
import dios

from saqc.constants import UNFLAGGED
from saqc.core.flags import Flags, History
from saqc.lib.tools import squeezeSequence, toSequence

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, Callable] = {}

FuncReturnT = Tuple[dios.DictOfSeries, Flags]


@dataclasses.dataclass
class CallState:
    func: Callable
    func_name: str

    data: DictOfSeries
    flags: Flags
    field: List[str]
    target: List[str] | None

    args: tuple
    kwargs: dict
    dec_kwargs: dict

    needs_squeezing: bool | None = None
    needs_masking: bool | None = None
    needs_demasking: bool | None = None
    mthresh: float | None = None
    mask: dios.DictOfSeries | None = None


def register(
    handles: Literal["data|flags", "index"] = "data|flags",
    datamask: Literal["all", "field"] | None = "all",
    multivariate: bool = False,
):

    # executed on module import

    if handles not in ["data|flags", "index"]:
        raise ValueError(
            f"invalid register decorator argument 'handles={repr(handles)}', "
            "only 'data|flags' or 'index' are allowed."
        )

    if datamask not in ["all", "field", None]:
        raise ValueError(
            f"invalid register decorator argument 'datamask={repr(datamask)}', "
            f"only None, 'all' or 'field' are allowed."
        )

    dec_kws = {"handles": handles, "datamask": datamask}

    def inner(func):
        func_name = func.__name__
        func.__multivariate__ = multivariate

        # executed if a register-decorated function is called,
        # nevertheless if it is called plain or via `SaQC.func`.
        @wraps(func)
        def callWrapper(data, field, flags, *args, **kwargs):

            state = CallState(
                func=func,
                func_name=func_name,
                data=data,
                flags=flags,
                field=toSequence(field),
                target=toSequence(kwargs.get("target", [])),
                args=args,
                kwargs=kwargs,
                dec_kwargs=dec_kws,
            )

            if handles == "index":
                # masking is possible, but demasking not,
                # because the index may changed, same apply
                # for squeezing the Flags
                state.needs_squeezing = False
                state.needs_demasking = False
            else:  # handles == "data|flags"
                state.needs_squeezing = True
                state.needs_demasking = True

            if datamask is None:
                # if we have nothing to mask, we aso have nothing to UNmask
                state.needs_masking = False
                state.needs_demasking = False
            else:
                state.needs_masking = True

            args, kwargs = _preCall(state)
            result = func(*args, **kwargs)
            return _postCall(result, state)

        FUNC_MAP[func_name] = callWrapper
        callWrapper._masking = datamask

        return callWrapper

    return inner


def _preCall(state: CallState):
    """
    Handler that runs before any call to a saqc-function.

    This is called before each call to a saqc-function, nevertheless if it is
    called via the SaQC-interface or plain by importing and direct calling.

    See Also
    --------
    _postCall: runs after a saqc-function call

    Returns
    -------
    args: tuple
        arguments to be passed to the actual call
    kwargs: dict
        keyword-arguments to be passed to the actual call
    """
    # set the masking threshold in kwargs that are passed to the function
    # but keep the call state's kwargs the original, because the threshold
    # is also stored in `state.mthresh`
    kwargs = state.kwargs.copy()
    mthresh = _getMaskingThresh(kwargs)
    kwargs["to_mask"] = mthresh
    state.mthresh = mthresh
    data, flags = state.data, state.flags

    if state.needs_demasking:
        datamask_kw = state.dec_kwargs["datamask"]
        # should we also mask a potential `target`?
        columns = _getMaskingColumns(state.data, state.field, datamask_kw)
        data, mask = _maskData(state.data, state.flags, columns, mthresh)
        state.mask = mask

    args = data, squeezeSequence(state.field), flags.copy(), *state.args

    return args, kwargs


def _postCall(result, old_state: CallState) -> FuncReturnT:
    """
    Handler that runs after any call to a saqc-function.

    This is called after a call to a saqc-function, nevertheless if it was
    called via the SaQC-interface or plain by importing and direct calling.

    Parameters
    ----------
    result : tuple
        the result from the called function, namely: data and flags

    old_state : dict
        control keywords from `_preCall`

    Returns
    -------
    data, flags : dios.DictOfSeries, saqc.Flags
    """
    data, flags = result

    if old_state.needs_squeezing:
        flags = _squeezeFlags(flags, old_state)

    if old_state.needs_demasking:
        data = _unmaskData(data, old_state)

    return data, flags


def _getMaskingColumns(
    data: dios.DictOfSeries, field: Sequence[str], datamask: str | None
) -> pd.Index:
    """
    Return columns to mask, by `datamask` (decorator keyword)

    Depending on the `datamask` kw, the following s returned:
        * None : empty pd.Index
        * 'all' : all columns from data
        * 'field': single entry Index

    Returns
    -------
    columns: pd.Index
        Data columns that need to be masked.

    Raises
    ------
    ValueError: if given datamask literal is not supported
    """
    if datamask is None:
        return pd.Index([])
    if datamask == "all":
        return data.columns
    if datamask == "field":
        return pd.Index(field)

    raise ValueError(f"wrong use of `register(datamask={repr(datamask)})`")


def _getMaskingThresh(kwargs):
    """
    Check the correct usage of the `to_mask` keyword, iff passed, otherwise return a default.

    Parameters
    ----------
    kwargs : dict
        The kwargs that will be passed to the saqc-function, possibly contain ``to_mask``.

    Returns
    -------
    threshold: float
        All data gets masked, if the flags are equal or worse than the threshold.

    Notes
    -----
    If ``to_mask`` is **not** in the kwargs, the threshold defaults to
     - ``-np.inf``
    If boolean ``to_mask`` is found in the kwargs, the threshold defaults to
     - ``-np.inf``, if ``True``
     - ``+np.inf``, if ``False``
    If a floatish ``to_mask`` is found in the kwargs, this value is taken as the threshold.
    """
    if "to_mask" not in kwargs:
        return UNFLAGGED

    thresh = kwargs["to_mask"]

    if not isinstance(thresh, (bool, float, int)):
        raise TypeError(f"'to_mask' must be of type bool or float")

    if thresh is True:  # masking ON
        thresh = UNFLAGGED

    if thresh is False:  # masking OFF
        thresh = np.inf

    thresh = float(thresh)  # handle int

    return thresh


def _isflagged(
    flagscol: Union[np.array, pd.Series], thresh: float
) -> Union[np.array, pd.Series]:
    """
    Return a mask of flags accordingly to `thresh`. Return type is same as flags.
    """
    if not isinstance(thresh, (float, int)):
        raise TypeError(f"thresh must be of type float, not {repr(type(thresh))}")

    if thresh == UNFLAGGED:
        return flagscol > UNFLAGGED

    return flagscol >= thresh


def _squeezeFlags(flags: Flags, old_state: CallState):
    """
    Generate flags from the temporary result-flags and the original flags.

    Parameters
    ----------
    flags : Flags
        The flags-frame, which is the result from a saqc-function

    old_state : CallState
        The state before the saqc-function was called

    Returns
    -------
    Flags
    """
    out = old_state.flags.copy()
    meta = {
        "func": old_state.func_name,
        "args": old_state.args,
        "kwargs": old_state.kwargs,
    }
    new_columns = flags.columns.difference(old_state.flags.columns)

    if old_state.dec_kwargs["datamask"] in (None, "all"):
        columns = flags.columns
    else:  # field or target
        columns = pd.Index(old_state.target or old_state.field)

    for col in columns.union(new_columns):

        if col not in out:  # ensure existence
            out.history[col] = History(index=flags.history[col].index)

        old_history = out.history[col]
        new_history = flags.history[col]

        # We only want to add columns, that were appended during the last function
        # call. If no such columns exist, we end up with an empty new_history.
        start = len(old_history.columns)
        new_history = _sliceHistory(new_history, slice(start, None))

        # NOTE:
        # Nothing to update -> i.e. a function did not set any flags at all.
        # This has implications for function writers: early returns out of
        # functions before `flags.__getitem__` was called once, make the
        # function call invisable to the flags/history machinery and likely
        # break translation schemes such as the `PositionalTranslator`
        if new_history.empty:
            continue

        squeezed = new_history.max(raw=True)
        out.history[col] = out.history[col].append(squeezed, meta=meta)

    return out


def _maskData(
    data: dios.DictOfSeries, flags: Flags, columns: Sequence[str], thresh: float
) -> Tuple[dios.DictOfSeries, dios.DictOfSeries]:
    """
    Mask data with Nans, if the flags are worse than a threshold.
        - mask only passed `columns` (preselected by `datamask`-kw from decorator)

    Returns
    -------
    masked : dios.DictOfSeries
        masked data, same dim as original
    mask : dios.DictOfSeries
        dios holding iloc-data-pairs for every column in `data`
    """
    mask = dios.DictOfSeries(columns=columns)

    # we use numpy here because it is faster
    for c in columns:
        col_mask = _isflagged(flags[c].to_numpy(), thresh)

        if col_mask.any():
            col_data = data[c].to_numpy(dtype=np.float64)

            mask[c] = pd.Series(col_data[col_mask], index=np.where(col_mask)[0])

            col_data[col_mask] = np.nan
            data[c] = col_data

    return data, mask


def _unmaskData(data: dios.DictOfSeries, old_state: CallState) -> dios.DictOfSeries:
    """
    Restore the masked data.

    Notes
    -----
    Even if this returns data, it works inplace !
    """
    # we have two options to implement this:
    #
    # =================================
    # set new data on old
    # =================================
    # col in old, in masked, in new:
    #    index differ : old <- new (replace column)
    #    else         : old <- new (set on masked: ``old[masked & new.notna()] = new``)
    # col in new only : old <- new (create column)
    # col in old only : old (delete column)
    #
    #
    # =================================
    # set old data on new (implemented)
    # =================================
    # col in old, in masked, in new :
    #    index differ : new (keep column)
    #    else         : new <- old (set on masked, ``new[masked & new.isna()] = old``)
    # col in new only : new (keep column)
    # col in old only : new (ignore, was deleted)

    # in old, in masked, in new
    columns = old_state.mask.columns.intersection(data.columns)

    for c in columns:

        # ignore
        if data[c].empty or old_state.mask[c].empty:
            continue

        # get the positions of values to unmask
        candidates = old_state.mask[c]
        # if the mask was removed during the function call, don't replace
        unmask = candidates[data[c].iloc[candidates.index].isna().to_numpy()]
        if unmask.empty:
            continue
        data[c].iloc[unmask.index] = unmask

    return data


def _sliceHistory(history: History, sl: slice) -> History:
    history.hist = history.hist.iloc[:, sl]
    history.meta = history.meta[sl]
    return history
