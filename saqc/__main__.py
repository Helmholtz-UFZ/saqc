#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa

from saqc.core import DictOfSeries
from saqc.core.core import TRANSLATION_SCHEMES
from saqc.parsing.reader import _ConfigReader
from saqc.version import __version__

logger = logging.getLogger("SaQC")
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s]: %(message)s"


def _setupLogging(loglvl):
    logger.setLevel(loglvl)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logging.basicConfig(level=loglvl, format=LOG_FORMAT)


def setupIO(nodata):
    reader = {
        ".csv": partial(pd.read_csv, index_col=0, parse_dates=True),
        ".parquet": pd.read_parquet,
    }

    writer = {
        ".csv": partial(pd.DataFrame.to_csv, header=True, index=True, na_rep=nodata),
        ".parquet": lambda df, outfile: pa.parquet.write_table(
            pa.Table.from_pandas(df), outfile
        ),
    }
    return reader, writer


def readData(reader_dict, fname):
    extension = Path(fname).suffix
    reader = reader_dict.get(extension)
    if not reader:
        raise ValueError(
            f"Unsupported file format '{extension}', use one of {tuple(reader_dict.keys())}"
        )
    return reader(fname)


def writeData(writer_dict, df, fname):
    extension = Path(fname).suffix
    writer = writer_dict.get(extension)
    if not writer:
        raise ValueError(
            f"Unsupported file format '{extension}', use one of {tuple(writer_dict.keys())}"
        )
    writer(df, fname)


@click.command()
@click.version_option(__version__)
@click.option(
    "-c",
    "--config",
    type=click.Path(),
    required=True,
    help="Path to a configuration file. Use a '.json' extension to provide a JSON-"
    "configuration. Otherwise files are treated as CSV.",
)
@click.option(
    "-d",
    "--data",
    type=click.Path(),
    multiple=True,
    required=True,
    help="Path to a data file.",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    required=False,
    help="Path to a output file.",
)
@click.option(
    "--scheme",
    default="simple",
    show_default=True,
    type=click.Choice(tuple(TRANSLATION_SCHEMES.keys())),
    help="A flagging scheme to use.",
)
@click.option(
    "--nodata", default=np.nan, help="Set a custom nodata value.", show_default=True
)
@click.option(
    "--log-level",
    "-ll",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    help="Set log verbosity.",
)
@click.option(
    "--json-field",
    default=None,
    help="Use the value from the given FIELD from the root object of a json file. The "
    "value must hold a array of saqc tests. If the option is not given, a passed "
    "JSON config is assumed to have an array of saqc tests as root element.",
)
def main(
    config: str,
    data: str,
    scheme: str,
    outfile: str,
    nodata: str | float,
    log_level: str,
    json_field: str | None,
):
    # data is always a list of data files

    _setupLogging(log_level)
    reader, writer = setupIO(nodata)
    data = [readData(reader, f) for f in data]

    config = str(config)
    cr = _ConfigReader(data=data, scheme=scheme)
    if config.endswith("json"):
        f = None
        if json_field is not None:
            f = lambda j: j[str(json_field)]
        cr = cr.readJson(config, unpack=f)
    else:
        cr = cr.readCsv(config)

    saqc = cr.run()

    data_result = saqc.data.to_pandas()
    flags_result = saqc.flags
    if isinstance(flags_result, DictOfSeries):
        flags_result = flags_result.to_pandas()

    if outfile:
        data_result.columns = pd.MultiIndex.from_product(
            [data_result.columns.tolist(), ["data"]]
        )

        if not isinstance(flags_result.columns, pd.MultiIndex):
            flags_result.columns = pd.MultiIndex.from_product(
                [flags_result.columns.tolist(), ["flags"]]
            )

        out = pd.concat([data_result, flags_result], axis=1).sort_index(
            axis=1, level=0, sort_remaining=False
        )

        writeData(writer, out, outfile)


if __name__ == "__main__":
    main()
