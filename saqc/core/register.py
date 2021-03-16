#!/usr/bin/env python
import logging
from typing import Dict, Optional, Union, Tuple, List
from typing_extensions import Literal
from functools import wraps
import dataclasses
import numpy as np
import pandas as pd
import dios
import warnings

from saqc.constants import *
from saqc.core.lib import SaQCFunction
from saqc.lib.types import FuncReturnT
from saqc.flagger.flags import Flagger, initFlagsLike

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, SaQCFunction] = {}

MaskingStrT = Literal["all", "field", "none"]


@dataclasses.dataclass
class CallCtrl:
    func: callable

    data: dios.DictOfSeries
    field: str
    flagger: Flagger

    args: tuple
    kwargs: dict

    masking: MaskingStrT = None
    mthresh: float = None
    mask: dios.DictOfSeries = None


def register(masking: MaskingStrT = "all", module: Optional[str] = None):

    # executed on module import
    def inner(func):
        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"

        # executed if a register-decorated function is called,
        # nevertheless if it is called plain or via `SaQC.func`.
        @wraps(func)
        def callWrapper(*args, **kwargs):
            args, kwargs, ctrl = _preCall(func, args, kwargs, masking, func_name)
            result = func(*args, **kwargs)
            return _postCall(result, ctrl, func_name)

        FUNC_MAP[func_name] = SaQCFunction(func_name, callWrapper)

        return callWrapper

    return inner


def _preCall(func: callable, args: tuple, kwargs: dict, masking: MaskingStrT, fname: str):
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
    ctrl: CallCtrl
        control keyword-arguments passed to `_postCall`

    """
    mthresh = _getMaskingThresh(masking, kwargs, fname)
    kwargs['to_mask'] = mthresh

    data, field, flagger, *args = args
    ctrl = CallCtrl(func, data.copy(), field, flagger.copy(), args, kwargs, masking=masking, mthresh=mthresh)

    # handle data - masking
    columns = _getMaskingColumns(data, field, masking)
    data, mask = _maskData(data, flagger, columns, mthresh)

    # store mask
    ctrl.mask = mask

    # handle flags - clearing
    flagger = _prepareFlags(flagger, masking)

    args = data, field, flagger, *args
    return args, kwargs, ctrl


def _postCall(result, ctrl: CallCtrl, fname: str) -> FuncReturnT:
    """
    Handler that runs after any call to a saqc-function.

    This is called after a call to a saqc-function, nevertheless if it was
    called via the SaQC-interface or plain by importing and direct calling.

    Parameters
    ----------
    result : tuple
        the result from the called function, namely: data and flagger

    ctrl : dict
        control keywords from `_preCall`

    fname : str
        Name of the (just) called saqc-function

    Returns
    -------
    data, flagger : dios.DictOfSeries, saqc.flagger.Flagger
    """
    data, flagger = result
    flagger = _restoreFlags(flagger, ctrl)
    data = _unmaskData(data, ctrl)
    return data, flagger


def _getMaskingColumns(data: dios.DictOfSeries, field: str, masking: MaskingStrT):
    """
    Returns
    -------
    columns: pd.Index
        Data columns that need to be masked.

    Raises
    ------
    ValueError: if given masking literal is not supported
    """
    if masking == 'all':
        return data.columns
    if masking == 'none':
        return pd.Index([])
    if masking == 'field':
        return pd.Index([field])

    raise ValueError(f"wrong use of `register(masking={ctrl.masking})`")


def _getMaskingThresh(masking, kwargs, fname):
    """
    Check the correct usage of the `to_mask` keyword, iff passed, otherwise return a default.

    Parameters
    ----------
    masking : str
        The function-scope masking keyword a saqc-function is decorated with.
    kwargs : dict
        The kwargs that will be passed to the saqc-function, possibly contain ``to_mask``.
    fname : str
        The name of the saqc-function to be called later (not here), to use in meaningful
         error messages

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
    if 'to_mask' not in kwargs:
        return UNFLAGGED

    thresh = kwargs['to_mask']

    if not isinstance(thresh, (bool, float, int)):
        raise TypeError(f"'to_mask' must be of type bool or float")

    if masking == 'none' and thresh not in (False, np.inf):
        # TODO: fix warning reference to docu
        warnings.warn(f"the saqc-function {fname!r} ignore masking and therefore does not evaluate the passed "
                      f"'to_mask'-keyword. Please refer to the documentation: TODO")

    if thresh is True:  # masking ON
        thresh = UNFLAGGED

    if thresh is False:  # masking OFF
        thresh = np.inf

    thresh = float(thresh)  # handle int

    return thresh


# TODO: this is heavily undertested
def _maskData(data, flagger, columns, thresh) -> Tuple[dios.DictOfSeries, dios.DictOfSeries]:
    """
    Mask data with Nans by flags worse that a threshold and according to masking keyword in decorator.
    """
    mask = dios.DictOfSeries(columns=columns)

    # we use numpy here because it is faster
    for c in columns:
        col_mask = _getMask(flagger[c].to_numpy(), thresh)

        if any(col_mask):
            col_data = data[c].to_numpy(dtype=np.float64)
            col_data[col_mask] = np.nan

            data[c] = col_data
            mask[c] = pd.Series(col_mask, index=data[c].index, dtype=bool)

    return data, mask


def _getMask(flags: Union[np.array, pd.Series], thresh: float) -> Union[np.array, pd.Series]:
    """
    Return a mask of flags accordingly to `thresh`. Return type is same as flags.
    """
    if thresh == UNFLAGGED:
        return flags > UNFLAGGED

    return flags >= thresh


def _prepareFlags(flagger: Flagger, masking) -> Flagger:
    """
    Clear flags before each call.
    """
    # Either the index or the columns itself changed
    if masking == 'none':
        return flagger

    return initFlagsLike(flagger, initial_value=UNTOUCHED)


def _restoreFlags(flagger: Flagger, ctrl: CallCtrl):
    if ctrl.masking == 'none':
        return flagger

    result = ctrl.flagger

    columns = flagger.columns
    # take field column and all possibly newly added columns
    if ctrl.masking == 'field':
        columns = columns.difference(ctrl.flagger.columns)
        columns = columns.append(pd.Index([ctrl.field]))

    for c in columns:
        # this implicitly squash the new-flagger history (RHS) to a single column, which than is appended to
        # the old history (LHS). The new-flagger history possibly consist of multiple columns, one for each
        # time flags was set to the flagger.
        result[c] = flagger[c]

    return result


# TODO: this is heavily undertested
def _unmaskData(data: dios.DictOfSeries, ctrl: CallCtrl) -> dios.DictOfSeries:
    """
    Restore the masked data.

    Notes
    -----
    Even if this returns data, it work inplace !
    """
    if ctrl.masking == 'none':
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

    old = ctrl  # this alias simplifies reading a lot
    columns = old.mask.columns.intersection(data.columns)  # in old, in masked, in new

    for c in columns:

        # ignore
        if old.data[c].empty or data[c].empty or old.mask[c].empty:
            continue

        # on index changed, we simply ignore the old data
        if not old.data[c].index.equals(data[c].index):
            continue

        restore_old_mask = old.mask[c].to_numpy() & data[c].isna().to_numpy()

        # we have nothing to restore
        if not any(restore_old_mask):
            continue

        # restore old values if no new are present
        v_old, v_new = old.data[c].to_numpy(), data[c].to_numpy()
        data.loc[:, c] = np.where(restore_old_mask, v_old, v_new)

    return data

