#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass

from typing import Optional, Any
from typing_extensions import Literal


@dataclass
class ColumnSelector:
    field: str
    target: str
    regex: bool


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
        return f"line: {self.lineno}\nexpression: {self.expression}" + super().errorMessage()


class SaQCFunction:

    def __init__(self, name, function, *args, **keywords):
        self.name = name
        self.func = function
        self.args = args
        self.keywords = keywords

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.func.__name__}"

    def bind(self, *args, **keywords):
        return SaQCFunction(
            self.name, self.func,
            *(self.args + args),
            **{**self.keywords, **keywords}
        )

    def __call__(self, data, field, flagger, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(data, field, flagger, *self.args, *args, **keywords)

    def errorMessage(self) -> str:
        return f"function: {self.name}\narguments: {self.args}\nkeywords: {self.keywords}"
