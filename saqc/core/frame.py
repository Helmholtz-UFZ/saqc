#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Hashable, Mapping

import pandas as pd
from fancy_collections import DictOfPandas


class DictOfSeries(DictOfPandas):
    _key_types = (str, int, float, tuple)
    _value_types = (pd.Series, pd.DataFrame)

    def __init__(self, *args, **kwargs):
        # data is needed to prevent an
        # AttributeError on repr during
        # Errors within __init__
        self.data = {}
        self._attrs = None
        super().__init__(*args, **kwargs)

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.
        """
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

    def astype(self, dtype: str | type) -> DictOfSeries:
        """
        Cast a DictOfSeries object to the specified ``dtype``

        Parameters
        ----------
        dtype: data type to cast the entire object to.

        Returns
        -------
        DictOfSeries
        """
        out = DictOfSeries()
        for k, v in self.data.items():
            out[k] = v.astype(dtype)
        return out


DictOfSeries.empty.__doc__ = """
Indicator whether DictOfSeries is empty.

True if DictOfSeries is entirely empty (no items) or all
items are empty themselves.

Notes
-----
To only check if DictOfSeries has no items use ``len`` or ``bool``
buildins.

Examples
--------
>>> from saqc import DictOfSeries
>>> di1 = DictOfSeries()
>>> di1.empty
True

A DictOfSeries is also considered empty if all items within it are empty

>>> di2 = DictOfSeries(a=pd.Series(dtype=float), b=pd.Series(dtype='O'))
>>> assert di2['a'].empty and di2['b'].empty
>>> di2.empty
True

To differentiate between a DictOfSeries with no items and a
DictOfSeries with empty items use the buildin functions
`len` or `bool`

>>> len(di1)
0
>>> bool(di1)
False
>>> len(di2)
2
>>> bool(di2)
True

Returns
-------
bool
"""

DictOfSeries.to_pandas.__doc__ = """
Transform DictOfSeries to a pandas.DataFrame.

Because a pandas.DataFrame can not handle data of different
length, but DictOfSeries can, the missing data is filled with
NaNs or is dropped, depending on the keyword `how`.

Parameters
----------
how : {'outer', 'inner'}, default 'outer'
    Defines how the resulting DataFrame index is generated.

    - ``outer`` : The resulting DataFrame index is the combination
        of all indices merged together. If a column misses values at
        new index locations, `NaN`'s are filled.
    - ``inner`` : Only indices that are present in all columns are used
        for the resulting index. Filling logic is not needed, but values
        are dropped, if a column has indices that are not known to all
        other columns.

Returns
-------
frame: pandas.DataFrame

Examples
--------
Missing data locations are filled with NaN's

>>> from saqc import DictOfSeries
>>> a = pd.Series(11, index=range(2))
>>> b = pd.Series(22, index=range(3))
>>> c = pd.Series(33, index=range(1,9,3))
>>> di = DictOfSeries(a=a, b=b, c=c)
>>> di   # doctest: +NORMALIZE_WHITESPACE
    a |     b |     c |
===== | ===== | ===== |
0  11 | 0  22 | 1  33 |
1  11 | 1  22 | 4  33 |
      | 2  22 | 7  33 |

>>> di.to_pandas()   # doctest: +NORMALIZE_WHITESPACE
      a     b     c
0  11.0  22.0   NaN
1  11.0  22.0  33.0
2   NaN  22.0   NaN
4   NaN   NaN  33.0
7   NaN   NaN  33.0

or is dropped if `how='inner'`

>>> di.to_pandas(how='inner')   # doctest: +NORMALIZE_WHITESPACE
    a   b   c
1  11  22  33
"""
