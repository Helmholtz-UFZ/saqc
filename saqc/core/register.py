#!/usr/bin/env python

from typing import Dict, Optional
from typing_extensions import Literal
from functools import wraps

from saqc.core.lib import SaQCFunction

# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP: Dict[str, SaQCFunction] = {}

MaskingStrT = Literal["all", "field", "none"]


def register(masking: MaskingStrT = "all", module: Optional[str] = None):

    # this is called once on module import
    def inner(func):
        func_name = func.__name__
        if module:
            func_name = f"{module}.{func_name}"
        FUNC_MAP[func_name] = SaQCFunction(func_name, masking, func)

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
    ctrl: dict
        control keyword-arguments passed to `_postCall`

    """
    ctrl = dict(
        func=func,
        args=args,
        kwargs=kwargs,
        masking=masking,
    )
    return args, kwargs, ctrl


def _postCall(result, ctrl: dict):
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
    data: dios.DictOfSeries
    flagger: saqc.flagger.Flagger
    """
    return result
