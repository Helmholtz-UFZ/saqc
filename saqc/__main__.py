#! /usr/bin/env python
# -*- coding: utf-8 -*-

import click

import numpy as np
import pandas as pd

from saqc.core import run
from saqc.flagger import CategoricalFlagger
from saqc.flagger.dmpflagger import DmpFlagger, FlagFields


FLAGGERS = {
    "numeric": CategoricalFlagger([-1, 0, 1]),
    "category": CategoricalFlagger(["NIL", "OK", "BAD"]),
    "dmp": DmpFlagger(),
}


@click.command()
@click.option(
    "-c", "--config", type=click.Path(exists=True), required=True, help="path to the configuration file",
)
@click.option(
    "-d", "--data", type=click.Path(exists=True), required=True, help="path to the data file",
)
@click.option("-o", "--outfile", type=click.Path(exists=False), help="path to the output file")
@click.option(
    "--flagger", default="category", type=click.Choice(FLAGGERS.keys()), help="the flagging scheme to use",
)
@click.option("--nodata", default=np.nan, help="nodata value")
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]), help="set output verbosity"
)
@click.option("--fail/--no-fail", default=True, help="whether to stop the program run on errors")
def main(config, data, flagger, outfile, nodata, log_level, fail):

    data = pd.read_csv(data, index_col=0, parse_dates=True,)

    data_result, flagger_result = run(
        config_file=config,
        flagger=FLAGGERS[flagger],
        data=data,
        nodata=nodata,
        log_level=log_level,
        error_policy="raise" if fail else "warn",
    )

    if outfile:
        flags = flagger_result.getFlags()
        flags_out = flags.where((flags.isnull() | flagger_result.isFlagged()), flagger_result.GOOD)

        if isinstance(flagger_result, DmpFlagger):
            flags = flagger_result._flags
            flags.loc[flags_out.index, (slice(None), FlagFields.FLAG)] = flags_out.values
            flags_out = flags

        if not isinstance(flags_out.columns, pd.MultiIndex):
            flags_out.columns = pd.MultiIndex.from_product([flags.columns, ["flag"]])

        data_result.columns = pd.MultiIndex.from_product([data_result.columns, ["data"]])

        # flags_out.columns = flags_out.columns.map("_".join)
        data_out = data_result.join(flags_out)
        data_out.sort_index(axis="columns").to_csv(outfile, header=True, index=True, na_rep=nodata)


if __name__ == "__main__":
    main()
