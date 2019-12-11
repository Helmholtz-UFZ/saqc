#! /usr/bin/env python
# -*- coding: utf-8 -*-

import click

import numpy as np
import pandas as pd

from saqc.core import runner
from saqc.flagger import CategoricalFlagger

FLAGGERS = {
    "number": CategoricalFlagger([-1, 0, 1]),
    "string": CategoricalFlagger(["NIL", "OK", "BAD"])
}

@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="path to the configuration file")
@click.option("-d", "--data", type=click.Path(exists=True), required=True, help="path to the data file")
@click.option("--flagger", default="string", type=click.Choice(FLAGGERS.keys()), help="the flagging scheme to use")
@click.option("--fail/--no-fail", default=True, help="whether to stop the program run on errors")
@click.option("--nodata", default=np.nan, help="the nodata value")
@click.option("--plot/--noplot", default=False, help="whether to plot the results")
def main(config, data, flagger, fail, nodata, plot):

    data = pd.read_csv(
        data,
        index_col=0,
        parse_dates=True,
    )

    data, flagger = runner(
        config_file=config,
        flagger=FLAGGERS[flagger],
        data=data,
        nodata=nodata,
        error_policy="raise" if fail else "warn"
    )

    # TODO: write file
    flags = flagger.getFlags()


if __name__ == "__main__":
    main()
