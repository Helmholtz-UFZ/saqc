#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
import inspect
from functools import wraps
from typing import *

import matplotlib as mpl
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    create_model,
    field_validator,
)
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated, Protocol, runtime_checkable

from saqc.lib.checking import (
    checkFields,
    checkFreqStr,
    checkNewFields,
    checkOffsetStr,
    checkSaQC,
)

EXTERNAL_FLAG = Union[str, float, int]


METHOD_LITERALS = Literal[
    "fagg",
    "bagg",
    "nagg",
    "froll",
    "broll",
    "nroll",
    "fshift",
    "bshift",
    "nshift",
    "match",
    "sshift",
    "mshift",
    "invert",
]

AGG_FUNC_LITERALS = Literal[
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "last",
    "first",
    "std",
    "var",
    "count",
    "sem",
    "linear",
    "time",
]


LINKAGE_STRING = Literal[
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
]


class ValidatePublicMembers:
    """
    Mixin that decorates public members of the mixed-in class with the signature validator
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr, value in cls.__dict__.items():
            # pass
            if callable(value) and attr[0] != "_":
                setattr(
                    cls,
                    attr,
                    validate_signature(value, QcMethodSignature),
                )


class _ComparativeABC(abc.ABCMeta):
    """
    Meta class that spawns constraint types as results from comparative dunder
    """

    # instantiate with: cmpT('int', (int,),{})
    def __gt__(self, other):
        return create_model(
            f"{self.__qualname__}>{other}",
            __base__=RootModel,
            root=(Annotated[self.__qualname__, Field(gt=other)], ...),
        )

    def __ge__(self, other):
        return create_model(
            f"{self.__qualname__}>={other}",
            __base__=RootModel,
            root=(Annotated[self.__qualname__, Field(ge=other)], ...),
        )

    def __lt__(self, other):
        return create_model(
            f"{self.__qualname__}<{other}",
            __base__=RootModel,
            root=(Annotated[self.__qualname__, Field(lt=other)], ...),
        )

    def __le__(self, other):
        return create_model(
            f"{self.__qualname__}<={other}",
            __base__=RootModel,
            root=(Annotated[self.__qualname__, Field(le=other)], ...),
        )

    def __getitem__(self, item):

        c = {}
        b = ["", ""]
        if isinstance(item[0], slice):
            c["gt"] = item[0].stop
            b[0] = "(" + str(c["gt"]) + ", "
        else:
            c["ge"] = item[0]
            b[0] = "[" + str(c["ge"]) + ", "
        if isinstance(item[1], slice):
            c["lt"] = item[1].start
            b[1] = str(c["lt"]) + ")"
        else:
            c["le"] = item[1]
            b[1] = str(c["le"]) + "]"
        return create_model(
            f"{self.__qualname__} in {b[0]}{b[1]}",
            __base__=RootModel,
            root=(Annotated[self.__qualname__, Field(**c)], ...),
        )


class ArbitrarySignature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def validate_signature(func, signature_model=ArbitrarySignature):
    """
    Decorator to on-the-fly build data models for qc method signatures and evaluate those.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function's signature
        empty_cast = lambda x, y: y if x == inspect._empty else x
        signature = inspect.signature(func)

        # parameterize signature model
        _parameters = signature.parameters
        _model_items = [
            (signature.parameters[name].annotation, signature.parameters[name].default)
            for name in _parameters
        ]
        _model_items = [
            (empty_cast(_a[0], "Any"), empty_cast(_a[1], "..")) for _a in _model_items
        ]
        model = create_model(
            func.__name__ + "Signature",
            __base__=signature_model,
            **dict(zip(_parameters, _model_items)),
        )

        # instantiate args/kwargs
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        _args = bound_arguments.arguments
        model.model_validate(_args, context=_args)
        return func(*args, **kwargs)

    return wrapper


class OptionalNone:
    pass


@runtime_checkable
class GenericFunction(Protocol):
    __name__: str
    __globals__: dict[str, Any]

    def __call__(
        self, *args: pd.Series
    ) -> pd.Series | pd.DataFrame | Any: ...  # pragma: no cover


@runtime_checkable
class CurveFitter(Protocol):
    def __call__(
        self, data: np.ndarray, *params: float
    ) -> np.ndarray: ...  # pragma: no cover


# Defining composed types


class NewSaQCFields(RootModel[Union[str, list[str]]]):
    """
    Type that can be an annotation to SaQC methods parameters and is checked against argument 'self' in those methods.
    Consistency is only ensured if `self` is validated (and thus defined) before the parameter being of `NewSaQCField` type.
    """

    @field_validator("root", mode="after")
    @classmethod
    def root_eval(cls, v: str | list[str], info: ValidationInfo) -> [str | list[str]]:
        if (
            isinstance(info.context, dict)
            and ("self" in info.context)
            and hasattr(info.context["self"], "columns")
        ):
            checkNewFields(v, info.context["self"].columns)
        else:
            raise ValueError(
                "Trying to validate newSaQCField type outside context of SaQC Method and/or Signature Context is not given. \n"
                'parameter "self" of type SaQC needs to be in the context to validate type newSaQCField.\n'
                f"Got Field-info: {info} \n"
                f"Got Field-context: {info.context}"
            )
        return v


class SaQCFields(RootModel[Union[str, list[str]]]):
    """
    Type that can be an annotation to SaQC methods parameters and is checked against argument 'self' in those methods.
    Consistency is only ensured if `self` is validated before `SaQCField` type.
    """

    @field_validator("root", mode="after")
    @classmethod
    def root_eval(cls, v: str | list[str], info: ValidationInfo) -> [str | list[str]]:
        if (
            isinstance(info.context, dict)
            and ("self" in info.context)
            and hasattr(info.context["self"], "columns")
        ):
            checkFields(v, info.context["self"].columns)
        else:
            raise ValueError(
                f"Trying to validate SaQCField type outside context of SaQC Method and/or Signature Context is not given. \n"
                f'parameter "self" of type SaQC needs to be in the context to validate type SaQCField.\n'
                f"Got Field-info: {info} \n"
                f"Got Field-context: {info.context}"
            )
        return v


class ArrayLike(RootModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: np.ndarray | pd.Series | pd.DataFrame


class OffsetLike(RootModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: pd.Timedelta | pd.DateOffset


class OffsetStr(RootModel):
    root: Annotated[str, AfterValidator(checkOffsetStr)]


class FreqStr(RootModel):
    root: Annotated[str, AfterValidator(checkFreqStr)]


class SaQC(RootModel):
    root: Annotated[Any, AfterValidator(checkSaQC)]


class CompT(RootModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: ArrayLike | float | int


# Relational annotation types:
Int = _ComparativeABC("int", (int,), {})
Float = _ComparativeABC("float", (float,), {})


# Base Model for qc method signatures. Add root validator for global consistency checks.
class QcMethodSignature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
