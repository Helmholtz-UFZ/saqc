#!/usr/bin/env python
from typing import Dict, Optional, Union, Tuple, Callable, Literal
from functools import wraps
import dataclasses
import numpy as np
import pandas as pd
import dios
import saqc.core.history

from saqc.constants import *
from saqc.core.lib import SaQCFunction
from saqc.core.flags import initFlagsLike, Flags, History

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, SaQCFunction] = {}

MaskingStrT = Literal["all", "field", "none"]
FuncReturnT = Tuple[dios.DictOfSeries, Flags]


@dataclasses.dataclass
class CallState:
    func: Callable
    func_name: str

    data: dios.DictOfSeries
    flags: Flags
    field: str

    args: tuple
    kwargs: dict

    masking: MaskingStrT
    mthresh: float
    mask: dios.DictOfSeries


def processing(module: Optional[str] = None):

    # executed on module import
    def inner(func):
        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"

        @wraps(func)
        def callWrapper(data, field, flags, *args, **kwargs):
            kwargs["to_mask"] = _getMaskingThresh(kwargs)
            return func(data, field, flags, *args, **kwargs)

        FUNC_MAP[func_name] = SaQCFunction(func_name, callWrapper)
        return callWrapper

    return inner


def flagging(masking: MaskingStrT = "all", module: Optional[str] = None):

    # executed on module import
    def inner(func):
        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"

        # executed if a register-decorated function is called,
        # nevertheless if it is called plain or via `SaQC.func`.
        @wraps(func)
        def callWrapper(data, field, flags, *args, **kwargs):
            args = data, field, flags, *args
            args, kwargs, old_state = _preCall(func, args, kwargs, masking, func_name)
            result = func(*args, **kwargs)
            return _postCall(result, old_state)

        FUNC_MAP[func_name] = SaQCFunction(func_name, callWrapper)
        callWrapper._module = module
        callWrapper._masking = masking

        return callWrapper

    return inner


def _preCall(
    func: Callable, args: tuple, kwargs: dict, masking: MaskingStrT, fname: str
):
    """
    Handler that runs before any call to a saqc-function.

    This is called before each call to a saqc-function, nevertheless if it is
    called via the SaQC-interface or plain by importing and direct calling.

    Parameters
    ----------
    func : callable
        the function, which is called after this returns. This is not called here!

    args : tuple
        args to the function

    kwargs : dict
        kwargs to the function

    masking : str
        a string indicating which columns in data need masking

    See Also
    --------
    _postCall: runs after a saqc-function call

    Returns
    -------
    args: tuple
        arguments to be passed to the actual call
    kwargs: dict
        keyword-arguments to be passed to the actual call
    state: CallState
        control keyword-arguments passed to `_postCall`

    """
    mthresh = _getMaskingThresh(kwargs)
    kwargs["to_mask"] = mthresh

    data, field, flags, *args = args

    # handle data - masking
    columns = _getMaskingColumns(data, field, masking)
    masked_data, mask = _maskData(data, flags, columns, mthresh)

    # store current state
    state = CallState(
        func=func,
        func_name=fname,
        data=data,
        flags=flags,
        field=field,
        args=args,
        kwargs=kwargs,
        masking=masking,
        mthresh=mthresh,
        mask=mask,
    )

    # handle flags - clearing
    prepped_flags = _prepareFlags(flags, masking)

    args = masked_data, field, prepped_flags, *args
    return args, kwargs, state


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
    flags = _restoreFlags(flags, old_state)
    data = _unmaskData(data, old_state)
    return data, flags


def _getMaskingColumns(data: dios.DictOfSeries, field: str, masking: MaskingStrT):
    """
    Return columns to mask, by `masking` (decorator keyword)

    Depending on the `masking` kw, the following s returned:
        * 'all' : all columns from data
        * 'None' : empty pd.Index
        * 'field': single entry Index

    Returns
    -------
    columns: pd.Index
        Data columns that need to be masked.

    Raises
    ------
    ValueError: if given masking literal is not supported
    """
    if masking == "all":
        return data.columns
    if masking == "none":
        return pd.Index([])
    if masking == "field":
        return pd.Index([field])

    raise ValueError(f"wrong use of `register(masking={masking})`")


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


# TODO: this is heavily undertested
def _maskData(
    data, flags, columns, thresh
) -> Tuple[dios.DictOfSeries, dios.DictOfSeries]:
    """
    Mask data with Nans, if the flags are worse than a threshold.

        - mask only passed `columns` (preselected by `masking`-kw from decorator)
        - copies data in any case

    Returns
    -------
    masked : dios.DictOfSeries
        masked data, same dim as original
    mask : dios.DictOfSeries
        boolean dios of same dim as `masked`. True, where data was masked, elsewhere False.
    """
    mask = dios.DictOfSeries(columns=columns)
    data = data.copy()

    # we use numpy here because it is faster
    for c in columns:
        col_mask = _isflagged(flags[c].to_numpy(), thresh)

        if col_mask.any():
            col_data = data[c].to_numpy(dtype=np.float64)
            col_data[col_mask] = np.nan

            data[c] = col_data
            mask[c] = pd.Series(col_mask, index=data[c].index, dtype=bool)

    return data, mask


def _isflagged(
    flagscol: Union[np.array, pd.Series], thresh: float
) -> Union[np.array, pd.Series]:
    """
    Return a mask of flags accordingly to `thresh`. Return type is same as flags.
    """
    if thresh == UNFLAGGED:
        return flagscol > UNFLAGGED

    return flagscol >= thresh


def _prepareFlags(flags: Flags, masking) -> Flags:
    """
    Prepare flags before each call. Always returns a copy.

    Currently this only clears the flags, but in future,
    this should be sliced the flags to the columns, that
    the saqc-function needs.

    Always return a copy of flags or a new flags-frame.
    """
    # Either the index or the columns itself changed
    if masking == "none":
        return flags.copy()

    return initFlagsLike(flags)


def _restoreFlags(flags: Flags, old_state: CallState):
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
    columns = flags.columns

    if old_state.masking == "all":
        pass

    # The function processed a copy of the original flags and may or may not added some
    # columns. So we take only new history columns and define new flags with it, which
    # are enriched with meta later
    elif old_state.masking == "none":
        flags = flags.copy(deep=False)

        for c in flags.columns:
            # if a new field (aka. variable) was inserted, we take the full history and
            # no slicing is needed, which is the hidden else-case.
            if c in old_state.flags.columns:
                l = len(old_state.flags.history[c].columns)
                flags.history[c] = _sliceHistory(flags.history[c], slice(l, None))

    # take field column and all possibly newly added columns
    elif old_state.masking == "field":
        columns = columns.difference(old_state.flags.columns)
        columns = columns.append(pd.Index([old_state.field]))

    else:
        raise RuntimeError(old_state.masking)

    out = old_state.flags.copy()

    # this implicitly squash the new flags history (RHS) to a single column, which than
    # is appended to the old history (LHS). Thus because the new flags history possibly
    # consists of multiple columns, one for each time a series or scalar was passed to
    # the flags.
    for c in columns:

        h = flags.history[c]
        hmax = h.max(raw=True)

        # # handle empty case (i.e. test didn't set any flags, can happen on early returns),
        # # to prevent a missing (empty) flags column
        # if h.empty:
        #     out.history[c] = h.copy()
        #     continue

        # # if nothing was touched we have no need to clutter the history
        # if (hmax == UNTOUCHED).all():
        #     continue

        out[c] = hmax

        # we enrich the (already existing !) empty meta with some infos
        history = out.history[c]
        history.meta[-1].update(
            {
                "func": old_state.func_name,
                "args": old_state.args,
                "keywords": old_state.kwargs,
            }
        )
        out.history[c] = history

    return out


# TODO: this is heavily undertested
def _unmaskData(data: dios.DictOfSeries, old_state: CallState) -> dios.DictOfSeries:
    """
    Restore the masked data.

    Notes
    -----
    Even if this returns data, it work inplace !
    """
    if old_state.masking == "none":
        return data

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

    columns = old_state.mask.columns.intersection(
        data.columns
    )  # in old, in masked, in new

    for c in columns:

        # ignore
        if old_state.data[c].empty or data[c].empty or old_state.mask[c].empty:
            continue

        restore_old_mask = old_state.mask[c].to_numpy() & data[c].isna().to_numpy()

        # we have nothing to restore
        if not restore_old_mask.any():
            continue

        # restore old values if no new are present
        old, new = old_state.data[c].to_numpy(), data[c].to_numpy()
        data.loc[:, c] = np.where(restore_old_mask, old, new)

    return data


def _sliceHistory(history: History, sl: slice) -> History:
    history.mask = history.mask.iloc[:, sl]
    history.hist = history.hist.iloc[:, sl]
    history.meta = history.meta[sl]
    return history
