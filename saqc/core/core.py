#! /usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd

from .reader import readConfig, prepareConfig, checkConfig
from .config import Fields
from .evaluator import evalExpression
from ..lib.plotting import plot_hook, plotall_hook
from ..flagger import BaseFlagger, CategoricalBaseFlagger, SimpleFlagger, DmpFlagger


def _collectVariables(meta, flagger, data, flags):
    """ find every requested variable in the meta, that is not already present in flags """
    if flags is None:
        ignore = list()
    else:
        ignore = list(flags.columns.get_level_values(level=0))

    varnames = []
    for idx, configrow in meta.iterrows():
        varname, assign = configrow[Fields.VARNAME], configrow[Fields.ASSIGN]
        if varname not in ignore and varname not in varnames:
            if varname in data or assign is True:
                varnames.append(varname)
    return varnames


def _checkInput(data, flags, flagger):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be of type pd.DataFrame')

    if isinstance(data.index, pd.MultiIndex):
        raise TypeError('the index of data is not allowed to be a multiindex')

    if isinstance(data.columns, pd.MultiIndex):
        raise TypeError('the columns of data is not allowed to be a multiindex')

    if not isinstance(flagger, BaseFlagger):
        flaggerlist = [CategoricalBaseFlagger, SimpleFlagger, DmpFlagger]
        raise TypeError(f'flagger must be of type {flaggerlist} or any inherit class from {BaseFlagger}')

    if flags is None:
        return

    if not isinstance(flags, pd.DataFrame):
        raise TypeError('flags must be of type pd.DataFrame')

    if isinstance(data.index, pd.MultiIndex):
        raise TypeError('the index of flags is not allowed to be a multiindex')

    if len(data) != len(flags):
        raise ValueError('the index of flags and data has not the same length')

    # do not test columns as they not necessarily must be the same


def _setup():
    pd.set_option('mode.chained_assignment', 'warn')


def runner(metafname, flagger, data, flags=None, nodata=np.nan, error_policy='raise'):
    _setup()
    _checkInput(data, flags, flagger)
    config = prepareConfig(readConfig(metafname), data)

    # split config into the test and some 'meta' data
    tests = config.filter(regex=Fields.TESTS)
    meta = config[config.columns.difference(tests.columns)]

    # prepapre the flags
    varnames = _collectVariables(meta, flagger, data, flags)
    fresh = flagger.initFlags(pd.DataFrame(index=data.index, columns=varnames))
    flags = fresh if flags is None else flags.join(fresh)

    # this checks comes late, but the compiling of the user-test need fully prepared flags
    checkConfig(config, data, flags, flagger, nodata)

    # the outer loop runs over the flag tests, the inner one over the
    # variables. Switching the loop order would complicate the
    # reference to flags from other variables within the dataset
    for _, testcol in tests.iteritems():

        # NOTE: just an optimization
        if testcol.dropna().empty:
            continue

        for idx, configrow in meta.iterrows():

            # store config params in some handy variables
            varname = configrow[Fields.VARNAME]
            start_date = configrow[Fields.START]
            end_date = configrow[Fields.END]

            flag_test = testcol[idx]
            if pd.isnull(flag_test):
                continue

            if varname not in data and varname not in flags:
                continue

            # prepare the data for the tests
            dchunk = data.loc[start_date:end_date]
            fchunk = flags.loc[start_date:end_date]
            if dchunk.empty:
                continue

            # actually run the tests
            try:
                dchunk, ffchunk = evalExpression(
                    flag_test,
                    data=dchunk, flags=fchunk.copy(), field=varname,
                    flagger=flagger, nodata=nodata)
            except Exception as e:
                if _handleErrors(e, configrow, flag_test, error_policy):
                    raise e
                else:
                    continue

            # write back the (new) flagged data
            data.loc[start_date:end_date] = dchunk
            flags.loc[start_date:end_date] = ffchunk.squeeze()

            plot_hook(dchunk, fchunk, ffchunk, varname, configrow[Fields.PLOT], flag_test, flagger)

        # NOTE: this method should be removed
        flagger.nextTest()

    plotall_hook(data, flags, flagger)

    return data, flags


def _handleErrors(err, configrow, test, policy):
    line = configrow[Fields.LINENUMBER]
    msg = f" config, line {line}, test: `{test}` failed with `{type(err).__name__}: {err}`"
    if policy == 'raise':
        return True
    elif policy == 'ignore':
        logging.debug(msg)
        return False
    elif policy == 'warn':
        logging.warning(msg)
        return False
    return True
