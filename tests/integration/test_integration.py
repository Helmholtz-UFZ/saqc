#!/usr/bin/env python
from click.testing import CliRunner
import os


def test__main__py():
    import saqc.__main__

    # if not run from project root
    projpath = os.path.dirname(saqc.__file__) + "/../"
    args = [
        "--config",
        projpath + "ressources/data/config_ci.csv",
        "--data",
        projpath + "ressources/data/data.csv",
        "--outfile",
        "/tmp/test.csv",  # the filesystem temp dir
    ]
    runner = CliRunner()

    for scheme in ["float", "positional", "dmp", "simple"]:
        result = runner.invoke(saqc.__main__.main, args + ["--scheme", scheme])
        assert result.exit_code == 0, result.output
