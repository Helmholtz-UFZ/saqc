#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Union

import saqc
from saqc.constants import UNFLAGGED, BAD
from saqc.lib.types import GenericFunction


class Generic:
    def genericProcess(
        self,
        field: str | Sequence[str],
        func: GenericFunction,
        target: str | Sequence[str] = None,
        flag: float = UNFLAGGED,
        to_mask: float = UNFLAGGED,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("genericProcess", locals())

    def genericFlag(
        self,
        field: Union[str, Sequence[str]],
        func: GenericFunction,
        target: Union[str, Sequence[str]] = None,
        flag: float = BAD,
        to_mask: float = UNFLAGGED,
        **kwargs,
    ) -> saqc.SaQC:
        return self._defer("genericFlag", locals())
