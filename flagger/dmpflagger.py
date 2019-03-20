#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .baseflagger import BaseFlagger


class DmpFlagger(BaseFlagger):
    def __init__(self, no_flag=0, flag=2):
        super().__init__(no_flag, flag)
        self.flag_fields = ["quality_flag", "quality_clause",
                            "quality_comment"]
        self.flag_lookup = {
            0: ["OK", "NIL", "NIL"],
            1: ["DOUBTFUL", "NIL", "NIL"],
            2: ["BAD", "NIL", "NIL"]}

    def emptyFlags(self, data, value=np.nan, **kwargs):
        columns = pd.MultiIndex.from_product([data.columns, self.flag_fields])
        return pd.DataFrame(data=value,
                            columns=columns, index=data.index,
                            dtype=str)

    def setFlag(self, flags, flag=None, **kwargs):
        if flag is None:
            flag = self.flag
        for field, f in zip(self.flag_fields, self.flag_lookup[flag]):
            flags.loc[:, (slice(None), field)] = f
        return flags
