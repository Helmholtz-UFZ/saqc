#! /usr/bin/env python
# -*- coding: utf-8 -*-

import click

import numpy as np
import pandas as pd
import logging

from saqc.core import SaQC
from saqc.flagger import CategoricalFlagger
from saqc.flagger.dmpflagger import DmpFlagger
import dios


logger = logging.getLogger("SaQC")

FLAGGERS = {
    "numeric": CategoricalFlagger([-1, 0, 1]),
    "category": CategoricalFlagger(["NIL", "OK", "BAD"]),
    "dmp": DmpFlagger(),
}


def _setup_logging(loglvl):
    logger.setLevel(loglvl)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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

    _setup_logging(log_level)

    # data = pd.read_csv(data, index_col=0, parse_dates=True,)
    import numpy as np
    data = np.random.uniform(0, 1, (1_000_000, 20))
    # data = np.random.uniform(0, 1, (10000, 20))
    data = pd.DataFrame(data=data, columns=np.arange(data.shape[1]).astype(str), index=pd.date_range(start="2000-01-01", periods=data.shape[0], freq="10Min"))
    data = dios.DictOfSeries(data)

    saqc = SaQC(flagger=FLAGGERS[flagger], data=data, nodata=nodata, error_policy="raise" if fail else "warn",)

    data_result, flagger_result = saqc.readConfig(config).getResult()

    return
    if outfile:
        data_result = data_result.to_df()
        flags = flagger_result.getFlags().to_df()
        flags_flagged = flagger_result.isFlagged().to_df()

        flags_out = flags.where((flags.isnull() | flags_flagged), flagger_result.GOOD)
        fields = {"data": data_result, "flags": flags_out}

        if isinstance(flagger_result, DmpFlagger):
            fields["comments"] = flagger_result.comments.to_df()
            fields["causes"] = flagger_result.causes.to_df()

        out = (
            pd.concat(fields.values(), axis=1, keys=fields.keys())
            .reorder_levels(order=[1, 0], axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )
        out.columns = out.columns.rename(["", ""])
        out.to_csv(outfile, header=True, index=True, na_rep=nodata)


if __name__ == "__main__":
    main()
