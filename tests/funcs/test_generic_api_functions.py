#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from dios.dios.dios import DictOfSeries

from saqc.constants import BAD, UNFLAGGED, FILTER_ALL
from saqc.core.flags import Flags
from saqc import SaQC
from saqc.core.register import _isflagged
from saqc.lib.tools import toSequence

from tests.common import initData


@pytest.fixture
def data():
    return initData()


def test_emptyData():
    # test that things do not break with empty data sets
    saqc = SaQC(data=pd.DataFrame({"x": [], "y": []}))

    saqc.flagGeneric("x", func=lambda x: x < 0)
    assert saqc.data.empty
    assert saqc.flags.empty

    saqc = saqc.processGeneric(field="x", target="y", func=lambda x: x + 2)
    assert saqc.data.empty
    assert saqc.flags.empty


def test_writeTargetFlagGeneric(data):
    params = [
        (["tmp"], lambda x, y: pd.Series(True, index=x.index.union(y.index))),
        (
            ["tmp1", "tmp2"],
            lambda x, y: [pd.Series(True, index=x.index.union(y.index))] * 2,
        ),
    ]
    for targets, func in params:
        expected_meta = {
            "func": "flagGeneric",
            "args": (),
            "kwargs": {
                "field": data.columns.tolist(),
                "func": func.__name__,
                "target": targets,
                "flag": BAD,
                "dfilter": FILTER_ALL,
            },
        }

        saqc = SaQC(data=data)
        saqc = saqc.flagGeneric(field=data.columns, target=targets, func=func, flag=BAD)
        for target in targets:
            assert saqc._flags.history[target].hist.iloc[0].tolist() == [BAD]
            assert saqc._flags.history[target].hist.iloc[0].tolist() == [BAD]
            assert saqc._flags.history[target].meta[0] == expected_meta


def test_overwriteFieldFlagGeneric(data):
    params = [
        (["var1"], lambda x: pd.Series(True, index=x.index)),
        (
            ["var1", "var2"],
            lambda x, y: [pd.Series(True, index=x.index.union(y.index))] * 2,
        ),
    ]

    flag = 12

    for fields, func in params:
        expected_meta = {
            "func": "flagGeneric",
            "args": (),
            "kwargs": {
                "field": fields,
                "target": fields,
                "func": func.__name__,
                "flag": flag,
                "dfilter": FILTER_ALL,
            },
        }

        saqc = SaQC(
            data=data.copy(),
            flags=Flags(
                {
                    k: pd.Series(data[k] % 2, index=data[k].index).replace(
                        {0: UNFLAGGED, 1: 127}
                    )
                    for k in data.columns
                }
            ),
        )

        res = saqc.flagGeneric(field=fields, func=func, flag=flag)
        for field in fields:
            assert (data[field] == res.data[field]).all(axis=None)
            histcol0 = res._flags.history[field].hist[0]
            histcol1 = res._flags.history[field].hist[1]
            assert (histcol1[histcol0 == 127.0].isna()).all()
            assert (histcol1[histcol0 != 127.0] == flag).all()
            assert res._flags.history[field].meta[0] == {}
            assert res._flags.history[field].meta[1] == expected_meta


def test_writeTargetProcGeneric(data):
    fields = ["var1", "var2"]
    params = [
        (["tmp"], lambda x, y: x + y),
        (["tmp1", "tmp2"], lambda x, y: (x + y, y * 2)),
    ]
    dfilter = 128
    for targets, func in params:

        expected_data = DictOfSeries(
            func(*[data[f] for f in fields]), columns=toSequence(targets)
        ).squeeze()

        expected_meta = {
            "func": "procGeneric",
            "args": (),
            "kwargs": {
                "field": data.columns.tolist(),
                "target": targets,
                "func": func.__name__,
                "flag": BAD,
                "dfilter": dfilter,
                "label": "generic",
            },
        }
        saqc = SaQC(
            data=data,
            flags=Flags(
                {k: pd.Series(127.0, index=data[k].index) for k in data.columns}
            ),
        )
        res = saqc.processGeneric(
            field=fields,
            target=targets,
            func=func,
            flag=BAD,
            dfilter=dfilter,
            label="generic",
        )
        assert (expected_data == res.data[targets].squeeze()).all(axis=None)
        # check that new histories where created
        for target in targets:
            assert res._flags.history[target].hist.iloc[0].tolist() == [BAD]
            assert res._flags.history[target].meta[0] == expected_meta


def test_overwriteFieldProcGeneric(data):
    params = [
        (["var1"], lambda x: x * 2),
        (["var1", "var2"], lambda x, y: (x + y, y * 2)),
    ]
    dfilter = 128
    flag = 12
    for fields, func in params:
        expected_data = DictOfSeries(
            func(*[data[f] for f in fields]), columns=fields
        ).squeeze()

        expected_meta = {
            "func": "procGeneric",
            "args": (),
            "kwargs": {
                "field": fields,
                "target": fields,
                "func": func.__name__,
                "flag": flag,
                "dfilter": dfilter,
                "label": "generic",
            },
        }

        saqc = SaQC(
            data=data,
            flags=Flags(
                {k: pd.Series(127.0, index=data[k].index) for k in data.columns}
            ),
        )

        res = saqc.processGeneric(
            field=fields, func=func, flag=flag, dfilter=dfilter, label="generic"
        )
        assert (expected_data == res.data[fields].squeeze()).all(axis=None)
        # check that the histories got appended
        for field in fields:
            assert (res._flags.history[field].hist[0] == 127.0).all()
            assert (res._flags.history[field].hist[1] == 12.0).all()
            assert res._flags.history[field].meta[0] == {}
            assert res._flags.history[field].meta[1] == expected_meta
