#!/usr/bin/env python
import logging
from typing import Dict, Optional, Union, Tuple, List
from typing_extensions import Literal
from functools import wraps
import dataclasses
import numpy as np
import pandas as pd
import dios

from saqc.common import *
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
    to_mask: List[float] = None
    mask: dios.DictOfSeries = None


def register(masking: MaskingStrT = "all", module: Optional[str] = None):

    # executed on module import
    def inner(func):

        # executed if a register-decorated function is called,
        # nevertheless if it is called plain or via `SaQC.func`.
        @wraps(func)
        def callWrapper(*args, **kwargs):
            args, kwargs, ctrl = _preCall(func, args, kwargs, masking)
            result = func(*args, **kwargs)
            return _postCall(result, ctrl)

        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"
        FUNC_MAP[func_name] = SaQCFunction(func_name, callWrapper)

        return callWrapper

    return inner


def _preCall(func: callable, args: tuple, kwargs: dict, masking: MaskingStrT):
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
    kwargs.setdefault('to_mask', None)
    data, field, flagger, *args = args

    ctrl = CallCtrl(func, data.copy(), field, flagger.copy(), args, kwargs, masking=masking)

    # masking
    ctrl.to_mask = _getToMask(ctrl)
    columns = _getMaskingColumns(ctrl, ctrl.masking)
    data, ctrl.mask = _maskData(data, flagger, columns, ctrl.to_mask)

    # flags
    flagger = _prepareFlags(flagger, ctrl)

    args = data, field, flagger, *args
    return args, kwargs, ctrl


def _postCall(result, ctrl: CallCtrl) -> FuncReturnT:
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

    Returns
    -------
    data, flagger : dios.DictOfSeries, saqc.flagger.Flagger
    """
    data, flagger = result
    flagger = _restoreFlags(flagger, ctrl)
    data = _unmaskData(data, ctrl)
    return data, flagger


def _getMaskingColumns(ctrl: CallCtrl, masking: MaskingStrT):
    """
    Raises
    ------
    ValueError: if given masking literal is not supported
    """
    if masking == 'all':
        return ctrl.data.columns
    if masking == 'none':
        return pd.Index([])
    if masking == 'field':
        return pd.Index([ctrl.field])

    raise ValueError(f"wrong use of `register(masking={ctrl.masking})`")


def _getToMask(ctrl):
    to_mask = ctrl.kwargs['to_mask']
    _warnForUnusedMasking(ctrl.masking, to_mask)

    if to_mask is None:
        to_mask = [UNFLAGGED]

    return to_mask


def _warnForUnusedMasking(masking, to_mask):
    # warn if the user explicitly pass `to_mask=..` to a function that is
    # decorated by `register(masking='none')`, by which `to_mask` is ignored
    if masking == 'none' and to_mask not in (None, []):
        # todo: see following message
        logging.warning("`to_mask` is given, but the test ignore masking. Please refer to the documentation: TODO")


# TODO: this is heavily undertested
def _maskData(data, flagger, columns, to_mask) -> Tuple[dios.DictOfSeries, dios.DictOfSeries]:
    """
    Mask data with Nans by flags, according to masking and to_mask.
    """
    mask = dios.DictOfSeries(columns=columns)

    # we use numpy here because it is faster
    for c in columns:
        col_mask = _getMask(flagger[c].to_numpy(), to_mask)

        if any(col_mask):
            col_data = data[c].to_numpy(dtype=np.float64)
            col_data[col_mask] = np.nan

            data[c] = col_data
            mask[c] = pd.Series(col_mask, index=data[c].index, dtype=bool)

    return data, mask


# todo: solve with outcome of #GL160
def _getMask(flags: Union[np.array, pd.Series], to_mask: list) -> Union[np.array, pd.Series]:
    """
    Return a mask of flags accordingly to `to_mask`.
    Return type is same as flags.
    """

    if isinstance(flags, pd.Series):
        mask = pd.Series(False, index=flags.index, dtype=bool)
    else:
        mask = np.zeros_like(flags, dtype=bool)

    for f in to_mask:
        mask |= flags == f

    return ~mask


def _prepareFlags(flagger: Flagger, ctrl: CallCtrl) -> Flagger:
    """
    Clear flags before each call.
    """
    # either the index or the columns itself changed
    if ctrl.masking == 'none':
        return flagger

    return initFlagsLike(flagger, initial_value=UNTOUCHED)


def _restoreFlags(flagger: Flagger, ctrl: CallCtrl):
    if ctrl.masking == 'none':
        ctrl.flagger = flagger

    else:
        columns = flagger.columns
        if ctrl.masking == 'field':
            columns = columns.difference(ctrl.flagger.columns)
            columns = columns.append(pd.Index([ctrl.field]))

        for c in columns:
            ctrl.flagger[c] = flagger[c]

    return ctrl.flagger


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

        if old.data[c].empty or data[c].empty or old.mask[c].empty:
            continue

        if old.data[c].index.equals(data[c].index):
            restore_old_val = old.mask[c].to_numpy() & data[c].isna().to_numpy()

            if any(restore_old_val):
                ol, nw = old.data[c].to_numpy(), data[c].to_numpy()
                data.loc[:, c] = np.where(restore_old_val, ol, nw)

    return data

