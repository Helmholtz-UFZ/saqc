#!/usr/bin/env python
from click.testing import CliRunner
from pathlib import Path


def test__main__py(tmp_path):
    import saqc.__main__

    # if not run from project root
    projpath = Path(saqc.__file__).parents[1]
    args = [
        "--config",
        Path(projpath, "ressources/data/config.csv"),
        "--data",
        Path(projpath, "ressources/data/data.csv"),
        "--outfile",
        Path(tmp_path, "test.csv"),  # the filesystem temp dir
    ]
    runner = CliRunner()
    for scheme in ["float", "positional", "dmp", "simple"]:
        result = runner.invoke(saqc.__main__.main, args + ["--scheme", scheme])
        assert result.exit_code == 0, result.output
