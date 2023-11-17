#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Collection, Iterable, Literal, TypeVar, get_origin

import numpy as np
import pandas as pd

T = TypeVar("T")


# ====================================================================
# `isSomething`-Checks: must not raise Exceptions by checking the value (but
# might rise Exceptions on wrong usage) and should return a boolean
# value
# ====================================================================
#
# Module should not have no saqc dependencies
#
def isBoolLike(obj: Any, optional: bool = False) -> bool:
    """Return True if obj is a boolean or one of the integers 0 or 1.
    If optional is True, `None` also is considered a valid boolean.
    """
    return (
        pd.api.types.is_bool(obj)
        or optional
        and obj is None
        or pd.api.types.is_integer(obj)
        and obj in [0, 1]
    )


def isFloatLike(obj: Any) -> bool:
    return pd.api.types.is_float(obj) or pd.api.types.is_integer(obj)


def isIterable(obj: Any) -> bool:
    if isinstance(obj, Iterable) or pd.api.types.is_iterator(obj):
        return True
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def isScalar(obj: Any, optional: bool = False) -> bool:
    return optional and obj is None or np.isscalar(obj)


def isCallable(obj: Any, optional: bool = False) -> bool:
    return optional and obj is None or callable(obj)


def isFixedFrequencyOffset(obj: Any):
    """Check if obj is a `pd.DateOffset` and have a fixed frequency.

    Motivation:
        pd.Timedelta always considered to have a fixed frequency, but
        a date-offset might or might not have a fixed frequency.
        Operations like `pd.Series.rolling` need a window with a fixed
        frequency, but other operations like `pd.Series.resample`, for
        example can handle any frequencies (fixed and non-fixed)

    This function return True if the object is a subclass of
    `pd.offsets.BaseOffset` and have a fixed frequency.
    """
    return isinstance(obj, pd.offsets.BaseOffset) and not obj.base.is_anchored()


def isFrequencyString(obj: Any, fixed_only=False) -> bool:
    if not isinstance(obj, str):
        return False
    try:
        offset = pd.tseries.frequencies.to_offset(obj)
        if fixed_only:
            return isFixedFrequencyOffset(offset)
        return True
    except ValueError:
        return False


def isTimedeltaString(obj: Any, allow_NaT: bool = False) -> bool:
    if not isinstance(obj, str):
        return False
    try:
        return pd.notna(pd.Timedelta(obj)) or allow_NaT
    except (ValueError, TypeError):
        return False


def isValidFrequency(obj: Any, allow_str: bool = True, fixed_only: bool = False):
    return (
        not fixed_only
        and isinstance(obj, pd.offsets.BaseOffset)
        or fixed_only
        and isFixedFrequencyOffset(obj)
        or allow_str
        and isFrequencyString(obj, fixed_only=fixed_only)
    )


def isValidFuncSelection(
    obj: Any,
    allow_callable: bool = True,
    allow_operator_str: bool = False,
    allow_trafo_str: bool = False,
):
    from saqc.parsing.environ import ENV_OPERATORS, ENV_TRAFOS

    return (
        allow_callable
        and callable(obj)
        or allow_operator_str
        and obj in ENV_OPERATORS.keys()
        or allow_trafo_str
        and obj in ENV_TRAFOS.keys()
    )


def isValidWindow(obj: Any, allow_int: bool = True, allow_str: bool = True) -> bool:
    return (
        isinstance(obj, pd.Timedelta)
        or isFixedFrequencyOffset(obj)
        or allow_int
        and pd.api.types.is_integer(obj)
        and isInBounds(obj, 0)
        or allow_str
        and isTimedeltaString(obj)
    )


def isValidChoice(value: T, choices: Collection[T]) -> bool:
    """Return if value is in choices.

    Raises
    ======
        TypeError: if choices is not a kind of collection.
    """
    if not isinstance(choices, Collection):
        raise TypeError("'choices' must be some kind of collection")
    return value in choices


def isInBounds(
    val: int | float,
    left: int | float = -np.inf,
    right: int | float = np.inf,
    closed: Literal["left", "right", "both", "neither"] = "left",
):
    """
    Check if a value is in a given interval.

    val :
        value to check

    left :
        The left or lower bound, defaults to `-inf`

    right :
        The right or upper bound, defaults to `+inf`

    closed : default "left"
        Defines where the interval has closed or open bounds, defaults to `"left"`.
        * `"left"`: to include left bound [left, right)
        * `"right"`: to include right bound (left, right]
        * `"both"`: closed interval [left, right]
        * `"neither"`: (default) open interval (left, right)
    """
    validateChoice(closed, "closed", ["left", "right", "both", "neither"])
    if closed == "neither":
        return left < val < right
    if closed == "left":
        return left <= val < right
    if closed == "right":
        return left < val <= right
    if closed == "both":
        return left <= val <= right


# ====================================================================
# Validation-functions:
# They should raise an Exceptions if conditions are not fulfilled and
# should return None.
# ====================================================================


def validateScalar(value, name: str, optional: bool = False):
    if not isScalar(value, optional=optional):
        raise ValueError(
            f"{name!r} must be a scalar{' or None' if optional else ''}, "
            f"not of type {type(value).__qualname__!r}"
        )


def validateCallable(func, name: str, optional: bool = False):
    if not isCallable(func, optional=optional):
        raise TypeError(
            f"{name!r} must be a callable{' or None' if optional else ''}, "
            f"not of type {type(func).__qualname__!r}"
        )


def _isLiteral(obj: Any) -> bool:
    # issubclass and isinstance does not work
    # for SpecialTypes, like Literal
    return get_origin(obj) == Literal


def validateChoice(value: T, name: str, choices: Collection[T] | type(Literal)):
    from saqc.lib.tools import extractLiteral

    if _isLiteral(choices):
        choices = extractLiteral(choices)
    if not isValidChoice(value, choices):
        raise ValueError(f"{name!r} must be one of {set(choices)}, not {value!r}")


def isIntOrInf(obj: int | float) -> bool:
    return pd.api.types.is_integer(obj) or pd.api.types.is_float(obj) and np.isinf(obj)


def validateValueBounds(
    value: int | float,
    name: str,
    left: int | float = -np.inf,
    right: int | float = np.inf,
    closed: Literal["left", "right", "both", "neither"] = "left",
    strict_int: bool = False,
):
    if (
        not pd.api.types.is_number(value)
        or not isInBounds(value, left, right, closed)
        or strict_int
        and not isIntOrInf(value)
    ):
        ival_str = dict(
            left="right-open interval [{}, {})",
            right="left-open interval ({}, {}]",
            both="closed interval [{}, {}]",
            neither="open interval ({}, {})",
        ).get(closed, "interval |{}, {}|")
        raise ValueError(
            f"{name!r} must be an int{'' if strict_int else ' or float'} "
            f"in the {ival_str.format(left, right)}, not {value!r}"
        )


def validateFraction(
    value: int | float,
    name: str,
    closed: Literal["left", "right", "both", "neither"] = "both",
):
    """Raise a ValueError if value is not in the interval |0, 1|"""
    return validateValueBounds(
        value, name, left=0, right=1, closed=closed, strict_int=False
    )


def validateFrequency(
    value: str | pd.offsets.BaseOffset | pd.Timedelta,
    name: str,
    allow_str: bool = True,
    fixed_only=False,
):
    # we might want to use checking.py in tools.py, so we use a
    # late import here, to avoid circular import errors
    from saqc.lib.tools import joinExt

    types = ["a Timedelta", "a BaseOffset"]
    if allow_str:
        types.append("an offset-string")
    msg = f"{name!r} must be {joinExt(', ', types, ' or ')}, not {value!r}"

    if not isValidFrequency(value, allow_str=allow_str, fixed_only=fixed_only):
        raise ValueError(msg)


def validateFuncSelection(
    value: Any,
    name: str = "func",
    allow_callable: bool = True,
    allow_operator_str: bool = False,
    allow_trafo_str: bool = False,
):
    """
    Validate Function selection to be either a Callable or a kex fro the environments Dictionaries.
    """
    from saqc.lib.tools import joinExt
    from saqc.parsing.environ import ENV_OPERATORS, ENV_TRAFOS

    is_valid = isValidFuncSelection(
        value,
        allow_callable=allow_callable,
        allow_trafo_str=allow_trafo_str,
        allow_operator_str=allow_operator_str,
    )

    msg_c = ["of type callable"] * allow_callable
    msg_op = [f"a string out of {ENV_OPERATORS.keys()}"] * allow_operator_str
    msg_tr = [f"a string out of {ENV_TRAFOS.keys()}"] * allow_trafo_str
    msg = joinExt(", ", msg_c + msg_op + msg_tr, " or ")
    if not is_valid:
        raise ValueError(f"Parameter '{name}' must be {msg}. Got '{value}' instead.")


def validateWindow(
    value: int | str | pd.offsets.BaseOffset | pd.Timedelta,
    name: str = "window",
    allow_int: bool = True,
    allow_str: bool = True,
    optional: bool = False,
    index: pd.Index | None = None,
):
    """
    Check if a `window` parameter is valid.

    Parameters
    ----------
    value :
        The value of the window to check.

    name :
        The name of the window variable to use in error messages.

    allow_int :
        If ``True``, integer windows are considered valid.
        Default is ``True``.

    allow_str :
        If ``True``, offset-string windows are considered valid.
        Default is ``True``.

    optional :
        If ``True``, allow window to be ``None``

    index :
        A pandas Index that is checked to be datetime-like if a
        datetime-like window is used. If `None` or if an integer window
        is used, this check is ignored. Default is `None`.
    """
    # we might want to use checking.py in tools.py, so we use a
    # late import here, to avoid circular import errors
    from saqc.lib.tools import joinExt

    # first ensure we're called correctly
    if index is not None and not isinstance(index, pd.Index):
        raise TypeError(
            f"'index' must be None or of type pd.Index, "
            f"not of type {type(index).__qualname__!r}"
        )

    types = ["a Timedelta", "a BaseOffset"]
    if allow_int:
        types.append("a positive integer")
    if allow_str:
        types.append("an offset-string")
    if optional:
        types.append("None")
    msg = f"{name!r} must be {joinExt(', ', types, ' or ')}, not {value!r}"

    if optional and value is None:
        return

    if not isValidWindow(value, allow_int=allow_int, allow_str=allow_str):
        # try to get a bit more detail for the error message
        if isinstance(value, str) and allow_str:
            try:
                if pd.isna(pd.Timedelta(value)):
                    raise ValueError("Timedelta conversion resulted in 'NaT'")
            except Exception as e:
                raise ValueError(
                    f"{name!r} is not a valid offset-string, because: " + str(e)
                ) from e
        raise ValueError(msg)

    if (
        index is not None
        and isinstance(value, (str, pd.offsets.BaseOffset, pd.Timedelta))
        and not pd.api.types.is_datetime64_any_dtype(index)
    ):
        raise ValueError(
            f"Data must have a datetime based index, if a time based {name!r} "
            f"is used, but data has an index of dtype {index.dtype}."
            + " Use an integer instead."
            if allow_int
            else ""
        )


def validateMinPeriods(
    value: int | None, name="min_periods", minimum=0, maximum=np.inf, optional=True
):
    """check if `min_periods` is in the right-open interval [minimum,maximum)"""
    if optional and value is None:
        return
    validateValueBounds(value, name=name, left=minimum, right=maximum, strict_int=True)
