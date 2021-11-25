#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Union

import saqc
from saqc.constants import UNFLAGGED, BAD, FILTER_ALL
from saqc.lib.types import GenericFunction


class Generic:
    def processGeneric(
        self,
        field: str | Sequence[str],
        func: GenericFunction,
        target: str | Sequence[str] = None,
        flag: float = UNFLAGGED,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("processGeneric", locals())

    def flagGeneric(
        self,
        field: Union[str, Sequence[str]],
        func: GenericFunction,
        target: Union[str, Sequence[str]] = None,
        flag: float = BAD,
        dfilter: float = FILTER_ALL,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("flagGeneric", locals())
