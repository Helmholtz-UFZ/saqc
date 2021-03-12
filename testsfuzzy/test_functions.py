#! /usr/bin/env python
# -*- coding: utf-8 -*-


from hypothesis import given, settings
from hypothesis.strategies import data, from_type

from saqc.core.register import FUNC_MAP
from testsfuzzy.init import MAX_EXAMPLES, functionKwargs


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(drawer=data())
def callWontBreak(drawer, func_name: str):
    func = FUNC_MAP[func_name]
    kwargs = drawer.draw(functionKwargs(func))

    # TODO: workaround until `flag` is explicitly exposed in signature
    flag = drawer.draw(from_type(float))
    kwargs.setdefault('flag', flag)

    func(**kwargs)


# breaks
# ------

# NOTE:
# needs a more elaborated test, as it calls into
# `changepoints.assignChangePointClusters`
def test_breaks_flagJumps():
    callWontBreak("breaks.flagJumps")


def test_breaks_flagIsolated():
    callWontBreak("breaks.flagIsolated")


def test_breaks_flagMissing():
    callWontBreak("breaks.flagMissing")


# constants
# ---------

def test_constats_flagConstats():
    callWontBreak("constants.flagConstants")


def test_constants_flagByVariance():
    callWontBreak("constants.flagByVariance")


# flagtools
# ---------

def test_flagtools_clearFlags():
    callWontBreak("flagtools.clearFlags")


def test_flagtools_forceFlags():
    callWontBreak("flagtools.clearFlags")


# NOTE:
# all of the following tests fail to sample data for `flag=typing.Any`
# with the new flagger in place this should be easy to fix
def test_flagtools_flagGood():
    callWontBreak("flagtools.flagGood")


def test_flagtools_flagUnflagged():
    callWontBreak("flagtools.flagUnflagged")


# NOTE: the problem is `mflag` which can be Any
# def test_flagtools_flagManual():
#     callWontBreak("flagtools.flagManual")


# outliers
# --------
#
# NOTE: needs a more elaborated test, I guess
# def test_outliers_flagByStray():
#     callWontBreak("outliers.flagByStray")


# NOTE: fails in a strategy, maybe `Sequence[ColumnName]`
# def test_outliers_flagMVScores():
#     callWontBreak("outliers.flagMVScores")


# NOTE:
# fails as certain combinations of frquency strings don't make sense
# a more elaborate test is needed
# def test_outliers_flagRaise():
#     callWontBreak("outliers.flagRaise")
#
#
# def test_outliers_flagMAD():
#     callWontBreak("outliers.flagMAD")
#
#
# def test_outliers_flagByGrubbs():
#     callWontBreak("outliers.flagByGrubbs")
#
#
# def test_outliers_flagRange():
#     callWontBreak("outliers.flagRange")


# NOTE: fails in a strategy, maybe `Sequence[ColumnName]`
# def test_outliers_flagCrossStatistic():
#     callWontBreak("outliers.flagCrossStatistic")
