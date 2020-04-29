#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
import pytest

from test.common import initData
from saqc.flagger import DmpFlagger

@pytest.fixture
def data():
    return initData(cols=1)


def parseComments(data):
    return np.array([json.loads(v)["comment"] for v in data.to_df().values.flatten()])


def test_initFlags(data):
    flagger = DmpFlagger().initFlags(data=data)
    assert (flagger._flags == flagger.UNFLAGGED).all(axis=None)
    assert (flagger._causes == "").all(axis=None)
    assert (flagger._comments == "").all(axis=None)


def test_setFlaggerOuter(data):

    flagger = DmpFlagger()

    field = data.columns[0]

    df = data[field].iloc[::2].to_frame()
    data_right = pd.DataFrame(data=df.values, columns=[field], index=df.index + pd.Timedelta("1Min"))
    data_left = data[field].to_frame()

    left = (flagger
            .initFlags(data=data_left)
            .setFlags(field=field, flag=flagger.BAD, comment="left", cause="left"))

    right = (flagger
             .initFlags(data=data_right)
             .setFlags(field, flag=flagger.GOOD, comment="right", cause="right"))

    merged = left.setFlagger(right, join="outer")

    assert (merged._flags.loc[data_right.index] == flagger.GOOD).all(axis=None)
    assert (merged._causes.loc[data_right.index] == "right").all(axis=None)
    assert np.all(parseComments(merged._comments.loc[data_right.index]) == "right")

    assert (merged._flags.loc[data_left.index] == flagger.BAD).all(axis=None)
    assert (merged._causes.loc[data_left.index] == "left").all(axis=None)
    assert np.all(parseComments(merged._comments.loc[data_left.index]) == "left")


def test_setFlaggerInner(data):

    flagger = DmpFlagger()

    field = data.columns[0]

    data_right = data[field].iloc[::2].to_frame()
    data_left = data[field].to_frame()

    left = (flagger
            .initFlags(data=data_left)
            .setFlags(field=field, flag=flagger.BAD, comment="left", cause="left"))

    right = (flagger
             .initFlags(data=data_right)
             .setFlags(field, flag=flagger.GOOD, comment="right", cause="right"))

    merged = left.setFlagger(right, join="inner").getFlags().to_df()
    assert (merged.index == data_right.index).all(axis=None)
    assert (merged == flagger.GOOD).all(axis=None)


def test_getFlaggerDrop(data):
    flagger = DmpFlagger().initFlags(data)
    with pytest.raises(TypeError):
        flagger.getFlags(field=data.columns, drop="var")

    field = data.columns[0]
    expected = data[data.columns.drop(field)].to_df()

    filtered = flagger.getFlagger(drop=field)

    assert (filtered._flags.columns == expected.columns).all(axis=None)
    assert (filtered._comments.columns == expected.columns).all(axis=None)
    assert (filtered._causes.columns == expected.columns).all(axis=None)

    assert (filtered._flags.to_df().index== expected.index).all(axis=None)
    assert (filtered._comments.to_df().index== expected.index).all(axis=None)
    assert (filtered._causes.to_df().index== expected.index).all(axis=None)

