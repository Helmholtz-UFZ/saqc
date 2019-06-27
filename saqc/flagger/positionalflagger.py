#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numbers import Number
from typing import Union, Sequence, Any, Optional

import numpy as np
import pandas as pd

from .baseflagger import BaseFlagger
from ..lib.tools import numpyfy, broadcastMany
from ..lib.types import ArrayLike


class PositionalFlagger(BaseFlagger):

    def __init__(self, no_flag=9, critical_flag=2):
        super().__init__(no_flag=no_flag, flag=critical_flag)
        self._flag_pos = 1
        self._initial_flag_pos = 1

    def nextTest(self):
        self._flag_pos += 1

    def setFlag(self,
                flags: ArrayLike,
                flag: Optional[int] = None,
                flagpos: Optional[int] = None,
                **kwargs: Any) -> np.ndarray:
        if flag is None:
            flag = self.flag
        if flagpos is None:
            flagpos = self._flag_pos
        return self._setFlags(flags, flag, flagpos)

    def isFlagged(self, flags, flag=None):
        maxflags = self._getMaxflags(flags[np.isfinite(flags)])
        if flag is None:
            return (pd.notnull(maxflags) & (maxflags != self.no_flag))
        return maxflags == flag

    def _getMaxflags(self, flags: pd.DataFrame,
                     exclude: Union[int, Sequence] = 0) -> np.ndarray:

        flagmax = np.max(np.array(flags))
        ndigits = int(np.ceil(np.log10(flagmax)))

        exclude = set(np.array(exclude).ravel())
        out = np.zeros_like(flags)

        for pos in range(ndigits):
            if pos not in exclude:
                out = np.maximum(out, self._getFlags(flags, pos))

        return out

    def _getFlags(self, flags: pd.DataFrame, pos: int) -> np.ndarray:

        flags = self._prepFlags(flags)
        pos = np.broadcast_to(np.atleast_1d(pos), flags.shape)

        ndigits = np.floor(np.log10(flags)).astype(np.int)
        idx = np.where(ndigits >= pos)

        out = np.zeros_like(flags)
        out[idx] = flags[idx] // 10**(ndigits[idx]-pos[idx]) % 10

        return out

    def _prepFlags(self, flags: ArrayLike) -> np.ndarray:
        out = numpyfy(flags)
        return out

    def _setFlags(self, flags: pd.DataFrame,
                  values: Union[pd.DataFrame, int], pos: int) -> np.ndarray:

        flags, pos, values = broadcastMany(flags, pos, values)

        out = flags.astype(np.float64)

        # right-pad 'flags' with zeros, to assure the
        # desired flag position is available
        ndigits = np.floor(np.log10(out)).astype(np.int)
        idx = (ndigits < pos)
        out[idx] *= 10**(pos[idx]-ndigits[idx])
        ndigits = np.log10(out).astype(np.int)

        out[idx] += 10**(ndigits[idx]-pos[idx]) * values[idx]

        return out.astype(np.int64)
