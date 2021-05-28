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
    result = runner.invoke(saqc.__main__.main, args)
    assert result.exit_code == 0, result.output

    # for scheme in ["float", "dmp", "positional"]:
    for scheme in ["float", "positional"]:
        result = runner.invoke(saqc.__main__.main, args + ["--scheme", scheme])
        assert result.exit_code == 0, result.output
