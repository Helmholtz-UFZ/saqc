#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy

import pandas as pd

from dios import DictOfSeries
from saqc.flagger.baseflagger import BaseFlagger, COMPARATOR_MAP
from saqc.flagger.dmpflagger import DmpFlagger
from saqc.lib.tools import toSequence


FLAGS = ("-1", "0", "1", "2")


class PositionalFlagger(BaseFlagger):
    def __init__(self):
        super().__init__(dtype=str)

    def setFlags(
            self,
            field: str,
            loc=None,
            position=-1,
            flag=None,
            force: bool = False,
            inplace: bool = False,
            with_extra=False,
            flag_after=None,
            flag_before=None,
            win_flag=None,
            **kwargs
    ):
        assertScalar("field", field, optional=False)

        # prepping
        flag = str(self.BAD if flag is None else flag)
        self.isValidFlag(flag, fail=True)
        out = self if inplace else deepcopy(self)
        out_flags = out._flags[field]

        idx = self.getFlags(field, loc).index
        mask = pd.Series(True, index=idx, dtype=bool)
        mask = mask.reindex_like(out_flags).fillna(False)

        # replace unflagged with the magic starter '9'
        out_flags = out_flags.str.replace(f"^{self.UNFLAGGED}", "9", regex=True)

        # bring all flags to the desired length
        # length = position # if position > 0 else out_flags.str.len().max()
        if position == -1:
            length = position = out_flags.str.len().max()
        else:
            length = position = position + 1
        out_flags = out_flags.str.pad(length + 1, fillchar=self.GOOD, side="right")

        # we rigorously overwrite existing flags
        new_flags = out_flags.str[position]
        new_flags.loc[mask] = flag

        # calc window flags
        if flag_after is not None or flag_before is not None:
            win_mask, win_flag = self._getWindowMask(field, mask, flag_after, flag_before, win_flag, flag, force)
            new_flags.loc[win_mask] = win_flag

        out._flags[field] = out_flags.str[:position] + new_flags + out_flags.str[position+1:]
        return out

    def isFlagged(self, field=None, loc=None, flag=None, comparator=">"):

        field = slice(None) if field is None else field
        flags = self._getMaxFlag(field, loc).astype(int)
        flags = flags.loc[:, field]

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

    def toDmpFlagger(self):
        self = PositionalFlagger().initFlags(flags=self._flags)
        dmp_flagger = DmpFlagger().initFlags(data=self._flags)
        flag_map = {
            self.BAD: dmp_flagger.BAD,
            self.SUSPICIOUS: dmp_flagger.SUSPICIOUS,
            self.GOOD: dmp_flagger.GOOD,
        }
        for pos_flag, dmp_flag in flag_map.items():
            loc = self.isFlagged(flag=pos_flag, comparator="==")
            dmp_flagger._flags.aloc[loc] = dmp_flag

        dmp_flagger._comments.loc[:] = self._flags.to_df().applymap(lambda v: json.dumps({"flag": v}))
        dmp_flagger._causes.loc[:] = "OTHER"
        return dmp_flagger

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

