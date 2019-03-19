#! /usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractproperty
from numbers import Number

import pandas as pd


class AbstractFlagger(ABC):

    @abstractproperty
    def no_flag(self):
        raise NotImplementedError

    @abstractproperty
    def critical_flag(self):
        raise NotImplementedError

    def setFlag(self, flags: pd.DataFrame, flag: Number) -> pd.DataFrame:
        flags.loc[:] = flag
        return flags

    def isFlagged(self, flags: pd.DataFrame) -> pd.DataFrame:
        return flags != self.no_flag

    def nextTest(self):
        pass

    def firstTest(self):
        pass
