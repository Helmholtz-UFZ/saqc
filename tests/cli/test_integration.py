#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from click.testing import CliRunner
from pathlib import Path


# tmp_path: pytest internal fixture
@pytest.mark.slow
def test__main__py(tmp_path):
    import saqc.__main__

    # if not run from project root
    projpath = Path(saqc.__file__).parents[1]
    args = [
        "--config",
        Path(projpath, "sphinxdoc/resources/data/config.csv"),
        "--data",
        Path(projpath, "sphinxdoc/resources/data/data.csv"),
        "--outfile",
        Path(tmp_path, "test.csv"),  # the filesystem temp dir
    ]
    runner = CliRunner()
    for scheme in ["float", "positional", "dmp", "simple"]:
        result = runner.invoke(saqc.__main__.main, args + ["--scheme", scheme])
        assert result.exit_code == 0, result.output
