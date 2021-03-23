#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import warnings
from functools import partial
from pathlib import Path

import click

import numpy as np
import pandas as pd
import pyarrow as pa

from saqc.constants import *
from saqc.core import SaQC


logger = logging.getLogger("SaQC")


SCHEMES = {
    None: None,
    "numeric": NotImplemented,
    "category": NotImplemented,
    "dmp": NotImplemented,
}


def _setup_logging(loglvl):
    logger.setLevel(loglvl)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def setupIO(nodata):
    reader = {
        ".csv"     : partial(pd.read_csv, index_col=0, parse_dates=True),
        ".parquet" : pd.read_parquet
    }

    writer = {
        ".csv" : partial(pd.DataFrame.to_csv, header=True, index=True, na_rep=nodata),
        ".parquet" : lambda df, outfile: pa.parquet.write_table(pa.Table.from_pandas(df), outfile)
    }
    return reader, writer


def readData(reader_dict, fname):
    extension = Path(fname).suffix
    reader = reader_dict.get(extension)
    if not reader:
        raise ValueError(f"Unsupported file format '{extension}', use one of {tuple(reader.keys())}")
    return reader(fname)


def writeData(writer_dict, df, fname):
    extension = Path(fname).suffix
    writer = writer_dict.get(extension)
    if not writer:
        raise ValueError(f"Unsupported file format '{extension}', use one of {tuple(writer.keys())}")
    writer(df, fname)


@click.command()
@click.option(
    "-c", "--config", type=click.Path(exists=True), required=True, help="path to the configuration file",
)
@click.option(
    "-d", "--data", type=click.Path(exists=True), required=True, help="path to the data file",
)
@click.option("-o", "--outfile", type=click.Path(exists=False), help="path to the output file")
@click.option(
    "--flagger", default=None, type=click.Choice(SCHEMES.keys()), help="the flagging scheme to use",
)
@click.option("--nodata", default=np.nan, help="nodata value")
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]), help="set output verbosity"
)
@click.option("--fail/--no-fail", default=True, help="whether to stop the program run on errors")
def main(config, data, flagger, outfile, nodata, log_level, fail):

    if SCHEMES[flagger] is NotImplemented:
        warnings.warn("flagger is currently not supported")

    _setup_logging(log_level)
    reader, writer = setupIO(nodata)

    data = readData(reader, data)

    saqc = SaQC(data=data, nodata=nodata, error_policy="raise" if fail else "warn",)

    data_result, flagger_result = saqc.readConfig(config).getResult(raw=True)

    if outfile:
        data_result = data_result.to_df()
        flags = flagger_result.toFrame()
        unflagged = (flags == UNFLAGGED) | flags.isna()
        flags[unflagged] = GOOD

        fields = {"data": data_result, "flags": flags}

        out = (
            pd.concat(fields.values(), axis=1, keys=fields.keys())
            .reorder_levels(order=[1, 0], axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )
        out.columns = out.columns.rename(["", ""])
        writeData(writer, out, outfile)


if __name__ == "__main__":
    main()
