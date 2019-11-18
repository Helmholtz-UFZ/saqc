#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ..common import initData
from saqc.flagger import BaseFlagger

import numpy as np
import pandas as pd


def test_flagOrder():
    flagger = BaseFlagger([-1, 0, 1])
    assert flagger.UNFLAGGED < flagger.GOOD
    assert flagger.UNFLAGGED < flagger.BAD
    assert flagger.GOOD < flagger.BAD
    assert flagger.BAD == flagger.BAD


def test_accesors():
    unflagged = -1
    good = 0
    bad = 1
    flagger = BaseFlagger([unflagged, good, bad])
    assert flagger.UNFLAGGED == unflagged
    assert flagger.GOOD == good
    assert flagger.BAD == bad


def test_isFlagged():
    flagger = BaseFlagger([-1, 0, 1])
    data = initData(cols=1).iloc[:50]
    flags = flagger.initFlags(data)
    flags.iloc[:10] = flagger.BAD
    flags.iloc[10:20] = flagger.GOOD

    checks = {
        (flagger.UNFLAGGED, "=="): flags == flagger.UNFLAGGED,
        (flagger.UNFLAGGED, ">="): flags >= flagger.UNFLAGGED,
        (flagger.GOOD, ">"): (flags == flagger.BAD),
        (flagger.GOOD, "<"): flags == flagger.UNFLAGGED,
        (flagger.GOOD, "<="): flags <= flagger.GOOD,
    }

    for (flag, comparator), right in checks.items():
        left = flagger.isFlagged(flags, flag=flag, comparator=comparator)
        assert np.all(left == right)

