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
from saqc.flagger.flags import Flagger

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

    masking: str = None
    to_mask: List[float] = None
    mask: dios.DictOfSeries = None


def register(masking: MaskingStrT = "all", module: Optional[str] = None):

    # this is called once on module import
    def inner(func):
        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"
        FUNC_MAP[func_name] = SaQCFunction(func_name, func)

        # this is called if a register-decorated function is called,
        # nevertheless if it is called plain or via `SaQC.func`.
        @wraps(func)
        def saqcWrapper(*args, **kwargs):
            args, kwargs, ctrl = _preCall(func, args, kwargs, masking)
            result = func(*args, **kwargs)
            return _postCall(result, ctrl)

        return saqcWrapper

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
    data, field, flagger, *args = args

    ctrl = CallCtrl(func, data.copy(), field, flagger.copy(), args, kwargs, masking=masking)
    ctrl.to_mask = _getToMask(ctrl)
    columns = _getMaskingColumns(ctrl)
    data, ctrl.mask = _maskData(data, flagger, columns, ctrl.to_mask)

    args = data, field, flagger, *args
    return args, kwargs, ctrl


def _getMaskingColumns(ctrl: CallCtrl):
    """
    Raises
    ------
    ValueError: if given masking literal is not supported
    """
    if ctrl.masking == 'all':
        return ctrl.data.columns
    if ctrl.masking == 'none':
        return pd.Index([])
    if ctrl.masking == 'field':
        return pd.Index([ctrl.field])

    raise ValueError(f"wrong use of `register(masking={ctrl.masking})`")


def _getToMask(ctrl):
    to_mask = ctrl.kwargs.setdefault('to_mask', None)
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

        if np.any(col_mask):
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

    return mask


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
    data = _unmaskData(data_old=ctrl.data, mask_old=ctrl.mask,
                       data_new=data, flagger_new=flagger,
                       to_mask=ctrl.to_mask)
    return data, flagger


# TODO: this is heavily undertested
def _unmaskData(data_old, mask_old, data_new, flagger_new, to_mask) -> dios.DictOfSeries:
    # NOTE:
    # we only need to respect columns, that were masked,
    # and are also still present in new data.
    # this throws out:
    #  - any newly assigned columns
    #  - columns that were excluded from masking
    columns = mask_old.columns.intersection(data_new.columns)

    for col in columns:

        if mask_old[col].empty or data_new[col].empty:
            continue

        was_masked = mask_old[col]
        is_masked = _getMask(flagger_new[col], to_mask)

        # if index changed we just go with the new data.
        # A test should use `register(masking='none')` if it changes
        # the index but, does not want to have all NaNs on flagged locations.
        if was_masked.index.equals(is_masked.index):
            mask = was_masked.to_numpy() & is_masked.to_numpy() & data_new[col].isna().to_numpy()

            # reapplying old values on masked positions
            if np.any(mask):
                data = np.where(mask, data_old[col].to_numpy(), data_new[col].to_numpy())
                data_new[col] = pd.Series(data=data, index=is_masked.index, dtype=data_old[col].dtype)

    return data_new
