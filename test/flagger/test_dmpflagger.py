#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
import pytest

from test.common import initData

DmpFlagger = NotImplemented
pytest.skip("DmpFlagger is deprecated.", allow_module_level=True)


@pytest.fixture
def data():
    return initData(cols=1)


@pytest.fixture
def data_4cols():
    return initData(cols=4)


def parseComments(data):
    return np.array([json.loads(v)["comment"] for v in data.to_df().values.flatten()])


def test_initFlags(data):
    flagger = DmpFlagger().initFlags(data=data)
    assert (flagger._flags == flagger.UNFLAGGED).all(axis=None)
    assert (flagger._causes == "").all(axis=None)
    assert (flagger._comments == "").all(axis=None)


def test_mergeFlaggerOuter(data):

    flagger = DmpFlagger()

    field = data.columns[0]

    data_left = data

    data_right = data.to_df()
    dates = data_right.index.to_series()
    dates[len(dates) // 2 :] += pd.Timedelta("1Min")
    data_right.index = dates
    data_right = data_right.to_dios()

    left = flagger.initFlags(data=data_left).setFlags(
        field=field, flag=flagger.BAD, cause="SaQCLeft", comment="testLeft"
    )

    right = flagger.initFlags(data=data_right).setFlags(
        field=field, flag=flagger.GOOD, cause="SaQCRight", comment="testRight"
    )

    merged = left.merge(right, join="outer")

    right_index = data_right[field].index.difference(data_left[field].index)
    assert (merged._flags.loc[right_index] == flagger.GOOD).all(axis=None)
    assert (merged._causes.loc[right_index] == "SaQCRight").all(axis=None)
    assert np.all(parseComments(merged._comments.loc[right_index]) == "testRight")

    left_index = data_left[field].index
    assert (merged._flags.loc[left_index] == flagger.BAD).all(axis=None)
    assert (merged._causes.loc[left_index] == "SaQCLeft").all(axis=None)
    assert np.all(parseComments(merged._comments.loc[left_index]) == "testLeft")


def test_mergeFlaggerInner(data):

    flagger = DmpFlagger()

    field = data.columns[0]

    data_left = data
    data_right = data.iloc[::2]

    left = flagger.initFlags(data=data_left).setFlags(
        field=field, flag=flagger.BAD, cause="SaQCLeft", comment="testLeft"
    )

    right = flagger.initFlags(data=data_right).setFlags(
        field=field, flag=flagger.GOOD, cause="SaQCRight", comment="testRight"
    )

    merged = left.merge(right, join="inner")

    assert (merged._flags[field].index == data_right[field].index).all()
    assert (merged._causes[field].index == data_right[field].index).all()
    assert (merged._comments[field].index == data_right[field].index).all()

    assert (merged._flags[field] == flagger.BAD).all()
    assert (merged._causes[field] == "SaQCLeft").all(axis=None)
    assert np.all(parseComments(merged._comments) == "testLeft")


def test_sliceFlaggerDrop(data):
    flagger = DmpFlagger().initFlags(data)
    with pytest.raises(TypeError):
        flagger.getFlags(field=data.columns, drop="var")

    field = data.columns[0]
    expected = data[data.columns.drop(field)].to_df()

    filtered = flagger.slice(drop=field)

    assert (filtered._flags.columns == expected.columns).all(axis=None)
    assert (filtered._comments.columns == expected.columns).all(axis=None)
    assert (filtered._causes.columns == expected.columns).all(axis=None)

    assert (filtered._flags.to_df().index == expected.index).all(axis=None)
    assert (filtered._comments.to_df().index == expected.index).all(axis=None)
    assert (filtered._causes.to_df().index == expected.index).all(axis=None)

