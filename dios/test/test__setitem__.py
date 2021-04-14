from .test_setup import *
import pytest

#
# s1 = pd.Series(range(10), index=range(10))
# s2 = pd.Series(range(5, 10), index=range(5, 10))
# s3 = pd.Series(range(1, 30, 2), index=range(1, 30, 2))
# s4 = pd.Series(np.linspace(7, 13, 9), index=range(3, 12))
# s1.name, s2.name, s3.name, s4.name = 'a', 'b', 'c', 'd'
# d1 = DictOfSeries(data=dict(a=s1.copy(), b=s2.copy(), c=s3.copy(), d=s4.copy()))
#
# blist = [True, False, False, True]
# b = pd.Series([True, False] * 5, index=[1, 2, 3, 4, 5] + [6, 8, 10, 12, 14])
# B = d1 > 5
#
#
#
#
# BLIST = [True, False, False, True]
#
# LISTIDXER = [['a'], ['a', 'c'], pd.Series(['a', 'c'])]
# BOOLIDXER = [pd.Series(BLIST), d1.copy() > 10]
# SLICEIDXER = [slice(None), slice(-3, -1), slice(-1, 3), slice(None, None, 3)]
# MULTIIDXER = []  # [d1 > 9, d1 != d1, d1 == d1]
# EMPTYIDEXER = [[], pd.Series(), slice(3, 3), slice(3, -1), DictOfSeries()]
#
# INDEXERS = LISTIDXER + BOOLIDXER + SLICEIDXER + MULTIIDXER + EMPTYIDEXER
#
#
