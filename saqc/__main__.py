#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from functools import partial
from pathlib import Path

import click

import numpy as np
import pandas as pd
import pyarrow as pa

from saqc.core.reader import fromConfig
from saqc.core.core import TRANSLATION_SCHEMES


logger = logging.getLogger("SaQC")


def _setupLogging(loglvl):
    logger.setLevel(loglvl)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
            f"Unsupported file format '{extension}', use one of {tuple(reader.keys())}"
        )
    return reader(fname)


def writeData(writer_dict, df, fname):
    extension = Path(fname).suffix
    writer = writer_dict.get(extension)
    if not writer:
        raise ValueError(
            f"Unsupported file format '{extension}', use one of {tuple(writer.keys())}"
        )
    writer(df, fname)


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(),
    required=True,
    help="path to the configuration file",
)
@click.option(
    "-d",
    "--data",
    type=click.Path(),
    multiple=True,
    required=True,
    help="path to the data file",
)
@click.option(
    "-o", "--outfile", type=click.Path(exists=False), help="path to the output file"
)
@click.option(
    "--scheme",
    default=None,
    type=click.Choice(tuple(TRANSLATION_SCHEMES.keys())),
    help="the flagging scheme to use",
)
@click.option("--nodata", default=np.nan, help="nodata value")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    help="set output verbosity",
)
def main(config, data, scheme, outfile, nodata, log_level):
    # data is always a list of data files

    _setupLogging(log_level)
    reader, writer = setupIO(nodata)

    _data = []
    for dfile in data:
        df = readData(reader, dfile)
        _data.append(df)
    data = _data

    saqc = fromConfig(
        config,
        data=data,
        scheme=TRANSLATION_SCHEMES[scheme or "simple"](),
    )

    data_result, flags_result = saqc.result.data, saqc.result.flags

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
