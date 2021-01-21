#! /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import pandas as pd

from saqc.flagger.baseflagger import BaseFlagger


class Flags(pd.CategoricalDtype):
    def __init__(self, flags):
        # NOTE: all flag schemes need to support
        #       at least 3 flag categories:
        #       * unflagged
        #       * good
        #       * bad
        assert len(flags) > 2
        super().__init__(flags, ordered=True)


class CategoricalFlagger(BaseFlagger):
    def __init__(self, flags):
        super().__init__(dtype=Flags(flags))
        self._categories = self.dtype.categories

    @property
    def UNFLAGGED(self):
        return self._categories[0]

    @property
    def GOOD(self):
        return self._categories[1]

    @property
    def BAD(self):
        return self._categories[-1]

    def isSUSPICIOUS(self, flag):
        return flag in self._categories[2:-1]
