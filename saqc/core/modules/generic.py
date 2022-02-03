#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Union

import saqc
from saqc.constants import UNFLAGGED, BAD, FILTER_ALL
from saqc.lib.types import GenericFunction
from sphinxdoc.scripts.templates import doc
import saqc.funcs


class Generic:
    @doc(saqc.funcs.generic.processGeneric.__doc__)
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

    @doc(saqc.funcs.generic.flagGeneric.__doc__)
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
