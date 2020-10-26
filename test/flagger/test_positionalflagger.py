#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from test.common import initData
from saqc.flagger import PositionalFlagger


@pytest.fixture
def data():
    return initData(cols=2)


def test_initFlags(data):
    flagger = PositionalFlagger().initFlags(data=data)
    assert (flagger.isFlagged() == False).all(axis=None)
    assert (flagger.flags == flagger.UNFLAGGED).all(axis=None)


def test_setFlags(data):
    flagger = PositionalFlagger().initFlags(data=data)

    field = data.columns[0]
    mask = np.zeros(len(data[field]), dtype=bool)
    mask[1:10:2] = True

    flagger = flagger.setFlags(field=field, loc=mask, flag=flagger.SUSPICIOUS)
    assert (flagger.flags.loc[mask, field] == "91").all(axis=None)
    assert (flagger.flags.loc[~mask, field] == "90").all(axis=None)

    flagger = flagger.setFlags(field=field, loc=~mask, flag=flagger.BAD)
    assert (flagger.flags.loc[~mask, field] == "902").all(axis=None)
    assert (flagger.flags.loc[mask, field] == "910").all(axis=None)

    assert (flagger.flags[data.columns[1]] == "-1").all(axis=None)


def test_isFlagged(data):
    flagger = PositionalFlagger().initFlags(data=data)
    field = data.columns[0]

    loc_sus = slice(1, 20, 2)
    flagger = flagger.setFlags(field=field, loc=loc_sus, flag=flagger.SUSPICIOUS)
    assert (flagger.isFlagged(field=field, comparator=">=", flag=flagger.SUSPICIOUS)[loc_sus] == True).all(axis=None)
    assert (flagger.isFlagged(field=field, comparator=">", flag=flagger.SUSPICIOUS) == False).all(axis=None)

    loc_bad = slice(1, 10, 2)
    flagger = flagger.setFlags(field=field, loc=loc_bad, flag=flagger.BAD)
    assert (flagger.isFlagged(field=field, comparator=">")[loc_sus] == True).all(axis=None)
    assert (flagger.isFlagged(field=field, comparator=">=", flag=flagger.BAD)[loc_bad] == True).all(axis=None)
    assert (flagger.isFlagged(field=field, comparator=">", flag=flagger.BAD) == False).all(axis=None)
