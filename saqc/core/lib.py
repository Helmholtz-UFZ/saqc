#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import functools

from typing import Optional, Any
from typing_extensions import Literal


class ColumnSelector:
    def __init__(self, field, target=None):
        self.field = field
        self.target = target or field

    def __repr__(self):
        return f"{self.__class__.__name__}({self.field})"


# TODO: this seems obsolete
@dataclass
class APIController:
    plot: bool

    def errorMessage(self):
        return ""


@dataclass
class ConfigController(APIController):
    lineno: Optional[int] = None
    expression: Optional[str] = None

    def errorMessage(self):
        return (
            f"line: {self.lineno}\nexpression: {self.expression}"
            + super().errorMessage()
        )


class SaQCFunction:
    def __init__(
        self,
        name="dummy",
        function=lambda data, _, flags, **kwargs: (data, flags),
        *args,
        **keywords,
    ):

        self.name = name
        self.func = function
        self.args = args
        self.keywords = keywords

    @property
    def __name__(self):
        return self.name

    def __repr__(self):
        args = ", ".join(self.args)
        kwargs = ", ".join([f"{k}={v}" for k, v in self.keywords.items()])
        string = ",".join(filter(None, [args, kwargs]))
        return f"{self.__class__.__name__}.{self.func.__name__}({string})"

    def bind(self, *args, **keywords):
        return SaQCFunction(
            self.name, self.func, *(self.args + args), **{**self.keywords, **keywords}
        )

    def __call__(self, data, field, flags, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(data, field, flags, *self.args, *args, **keywords)

    def errorMessage(self) -> str:
        return (
            f"function: {self.name}\narguments: {self.args}\nkeywords: {self.keywords}"
        )
