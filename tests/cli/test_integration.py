#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest
from click.testing import CliRunner

FLOAT = [
    ",Battery,Battery,SM1,SM1,SM2,SM2\n",
    ",data,flags,data,flags,data,flags\n",
    "Date,,,,,,\n",
    "2016-04-01 00:00:00,nan,nan,nan,nan,29.3157,-inf\n",
    "2016-04-01 00:05:48,3573.0,-inf,32.685,-inf,nan,nan\n",
    "2016-04-01 00:15:00,nan,nan,nan,nan,29.3157,-inf\n",
    "2016-04-01 00:20:42,3572.0,-inf,32.7428,-inf,nan,nan\n",
    "2016-04-01 00:30:00,nan,nan,nan,nan,29.3679,255.0\n",
    "2016-04-01 00:35:37,3572.0,-inf,32.6186,-inf,nan,nan\n",
    "2016-04-01 00:45:00,nan,nan,nan,nan,29.3679,-inf\n",
]

SIMPLE = [
    ",Battery,Battery,SM1,SM1,SM2,SM2\n",
    ",data,flags,data,flags,data,flags\n",
    "Date,,,,,,\n",
    "2016-04-01 00:00:00,nan,nan,nan,nan,29.3157,UNFLAGGED\n",
    "2016-04-01 00:05:48,3573.0,UNFLAGGED,32.685,UNFLAGGED,nan,nan\n",
    "2016-04-01 00:15:00,nan,nan,nan,nan,29.3157,UNFLAGGED\n",
    "2016-04-01 00:20:42,3572.0,UNFLAGGED,32.7428,UNFLAGGED,nan,nan\n",
    "2016-04-01 00:30:00,nan,nan,nan,nan,29.3679,BAD\n",
    "2016-04-01 00:35:37,3572.0,UNFLAGGED,32.6186,UNFLAGGED,nan,nan\n",
    "2016-04-01 00:45:00,nan,nan,nan,nan,29.3679,UNFLAGGED\n",
]

POSITIONAL = [
    ",Battery,Battery,SM1,SM1,SM2,SM2\n",
    ",data,flags,data,flags,data,flags\n",
    "Date,,,,,,\n",
    "2016-04-01 00:00:00,nan,-9999,nan,-9999,29.3157,90000\n",
    "2016-04-01 00:05:48,3573.0,9,32.685,90,nan,-9999\n",
    "2016-04-01 00:15:00,nan,-9999,nan,-9999,29.3157,90000\n",
    "2016-04-01 00:20:42,3572.0,9,32.7428,90,nan,-9999\n",
    "2016-04-01 00:30:00,nan,-9999,nan,-9999,29.3679,90002\n",
    "2016-04-01 00:35:37,3572.0,9,32.6186,90,nan,-9999\n",
    "2016-04-01 00:45:00,nan,-9999,nan,-9999,29.3679,90000\n",
]

DMP = [
    ",Battery,Battery,Battery,Battery,SM1,SM1,SM1,SM1,SM2,SM2,SM2,SM2\n",
    ",data,quality_flag,quality_cause,quality_comment,data,quality_flag,quality_cause,quality_comment,data,quality_flag,quality_cause,quality_comment\n",
    "Date,,,,,,,,,,,,\n",
    "2016-04-01 00:00:00,nan,nan,nan,nan,nan,nan,nan,nan,29.3157,NIL,,\n",
    "2016-04-01 00:05:48,3573.0,NIL,,,32.685,NIL,,,nan,nan,nan,nan\n",
    "2016-04-01 00:15:00,nan,nan,nan,nan,nan,nan,nan,nan,29.3157,NIL,,\n",
    "2016-04-01 00:20:42,3572.0,NIL,,,32.7428,NIL,,,nan,nan,nan,nan\n",
    '2016-04-01 00:30:00,nan,nan,nan,nan,nan,nan,nan,nan,29.3679,BAD,OTHER,"{""test"": ""flagMAD"", ""comment"": """"}"\n',
    "2016-04-01 00:35:37,3572.0,NIL,,,32.6186,NIL,,,nan,nan,nan,nan\n",
    "2016-04-01 00:45:00,nan,nan,nan,nan,nan,nan,nan,nan,29.3679,NIL,,\n",
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "scheme, expected",
    [("float", FLOAT), ("simple", SIMPLE), ("positional", POSITIONAL), ("dmp", DMP)],
)
def test__main__py(tmp_path, scheme, expected):
    import saqc.__main__

    # if not run from project root
    projpath = Path(saqc.__file__).parents[1]
    outfile = Path(tmp_path, "test.csv")  # the filesystem's temp dir
    args = [
        "--config",
        Path(projpath, "docs/resources/data/config.csv"),
        "--data",
        Path(projpath, "docs/resources/data/data.csv"),
        "--outfile",
        outfile,
        "--scheme",
        scheme,
    ]
    result = CliRunner().invoke(saqc.__main__.main, args)
    assert result.exit_code == 0, result.output
    with open(outfile, "r") as f:
        result = f.readlines()[:10]
        assert result == expected
