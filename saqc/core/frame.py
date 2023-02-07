#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-

import warnings
from typing import List

from dios import DictOfSeries, to_dios  # noqa


def mergeDios(left: DictOfSeries, right: DictOfSeries, subset=None, join="merge"):
    # use dios.merge() as soon as it implemented
    # see https://git.ufz.de/rdm/dios/issues/15

    merged = left.copy()
    if subset is not None:
        right_subset_cols = right.columns.intersection(subset)
    else:
        right_subset_cols = right.columns

    shared_cols = left.columns.intersection(right_subset_cols)

    for c in shared_cols:
        l, r = left[c], right[c]
        if join == "merge":
            # NOTE:
            # our merge behavior is nothing more than an
            # outer join, where the right join argument
            # overwrites the left at the shared indices,
            # while on a normal outer join common indices
            # hold the values from the left join argument
            r, l = l.align(r, join="outer")
        else:
            l, r = l.align(r, join=join)
        merged[c] = l.combine_first(r)

    newcols = right_subset_cols.difference(left.columns)
    for c in newcols:
        merged[c] = right[c].copy()

    return merged


def concatDios(data: List[DictOfSeries], warn: bool = True, stacklevel: int = 2):
    # fast path for most common case
    if len(data) == 1 and data[0].columns.is_unique:
        return data[0]

    result = DictOfSeries()
    for di in data:
        for c in di.columns:
            if c in result.columns:
                if warn:
                    warnings.warn(
                        f"Column {c} already exist. Data is overwritten. "
                        f"Avoid duplicate columns names over all inputs.",
                        stacklevel=stacklevel,
                    )
            result[c] = di[c]

    return result
