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


@dataclass
class APIController:
    masking: Literal["none", "field", "all"]
    plot: bool
    to_mask: Any = None  # flagger.FLAG constants or a list of those

    def errorMessage(self):
        return f"masking: {self.masking}\nto_mask: {self.to_mask}"


@dataclass
class ConfigController(APIController):
    lineno: Optional[int] = None
    expression: Optional[str] = None

    def errorMessage(self):
        return f"line: {self.lineno}\nexpression: {self.expression}" + super().errorMessage()


class SaQCFunction:

    def __init__(self, name, masking, function, *args, **keywords):
        self.name = name
        self.masking = masking
        self.func = function
        self.args = args
        self.keywords = keywords

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.func.__name__}"

    def bind(self, *args, **keywords):
        return SaQCFunction(
            self.name, self.masking, self.func,
            *(self.args + args), **{**self.keywords, **keywords}
        )

    def __call__(self, data, field, flagger, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(data, field, flagger, *self.args, *args, **keywords)

    def errorMessage(self) -> str:
        return f"function: {self.name}\narguments: {self.args}\nkeywords: {self.keywords}"
