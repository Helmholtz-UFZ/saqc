#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Collection,
    List,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

CLOSURE_TO_NOTION = {
    None: "interval ({}, {})",
    "left": "right-open interval [{}, {})]",
    "right": "left-open interval ({}, {}]",
    "both": "closed interval [{}, {}]",
}


class ParameterOutOfBounds(Exception):
    def __init__(
        self,
        value: int | float,
        para_name: str,
        bounds: Tuple[str],
        closed: Literal["right", "left", "both"] = None,
    ):
        Exception.__init__(self)
        self.value = value
        self.para_name = para_name
        self.bounds = bounds
        self.closed = closed
        self.msg = "Parameter '{}' has to be in the {}, but {} was passed."

    def __str__(self):
        return self.msg.format(
            self.para_name,
            CLOSURE_TO_NOTION[self.closed].format(self.bounds[0], self.bounds[1]),
            self.value,
        )
