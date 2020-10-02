#! /usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import pandas as pd

from dios import DictOfSeries
from saqc.flagger.baseflagger import BaseFlagger, COMPARATOR_MAP
from saqc.lib.tools import assertScalar, toSequence


FLAGS = ("-1", "0", "1", "2")


class PositionalFlagger(BaseFlagger):
    def __init__(self):
        super().__init__(dtype=str)

    def setFlags(self, field, loc, position=-1, flag=None, force=False, inplace=False, **kwargs):
        assertScalar("field", field, optional=False)

        # prepping
        flag = str(self.BAD if flag is None else flag)
        self.isValidFlag(flag, fail=True)
        out = self if inplace else deepcopy(self)
        out_flags = out._flags[field]

        # replace unflagged with the magic starter '9'
        out_flags = out_flags.str.replace(f"^{self.UNFLAGGED}", "9", regex=True)

        # bring all flags to the desired length
        # length = position # if position > 0 else out_flags.str.len().max()
        if position == -1:
            length = position = out_flags.str.len().max()
        else:
            length = position = position + 1
        out_flags = out_flags.str.pad(length + 1, fillchar=self.GOOD, side="right")

        # we rigerously overwrite existing flags 
        new_flags = out_flags.str[position]
        new_flags[loc] = flag

        out._flags[field] = out_flags.str[:position] + new_flags + out_flags.str[position+1:]
        return out

    def isFlagged(self, field=None, loc=None, flag=None, comparator=">"):

        flags = self._getMaxFlag(field, loc).astype(int)

        # notna() to prevent nans to become True,
        # eg.: `np.nan != 0 -> True`
        flagged = flags.notna()
        flags_to_compare = set(toSequence(flag, self.GOOD))
        if not flags_to_compare:
            flagged[:] = False
            return flagged

        cp = COMPARATOR_MAP[comparator]
        for f in flags_to_compare:
            self.isValidFlag(f, fail=True)
            flagged &= cp(flags, int(f))
        return flagged

    def isValidFlag(self, flag, fail=False):
        check = flag in FLAGS
        if check is False and fail is True:
            raise ValueError(f"invalid flag {flag}, given values should be in '{FLAGS}'")
        return check

    def _getMaxFlag(self, field, loc):

        data = {}
        flags = self.getFlags(field, loc)
        if isinstance(flags, pd.Series):
            flags = flags.to_frame()
        for col_name, col in flags.iteritems():
            mask = col != self.UNFLAGGED
            col = col.str.replace("^9", "0", regex=True)
            col[mask] = col[mask].apply(lambda x: max(list(x)))
            data[col_name] = col
        return DictOfSeries(data)

    @property
    def UNFLAGGED(self):
        return FLAGS[0]

    @property
    def GOOD(self):
        return FLAGS[1]

    @property
    def SUSPICIOUS(self):
        return FLAGS[2]

    @property
    def BAD(self):
        return FLAGS[3]

    def isSUSPICIOUS(self, flag):
        return flag == self.SUSPICIOUS

