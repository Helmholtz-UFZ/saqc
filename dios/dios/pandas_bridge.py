#!/usr/bin/env python


__author__ = "Bert Palm"
__email__ = "bert.palm@ufz.de"
__copyright__ = "Copyright 2020, Helmholtz-Zentrum für Umweltforschung GmbH - UFZ"


from pandas.core.common import (
    is_bool_indexer,
    is_null_slice,
)
from pandas.core.dtypes.common import (
    is_nested_list_like,
)
from pandas.api.types import (
    is_list_like,
    is_hashable,
    is_integer,
    is_dict_like,
    is_scalar,
    # Unlike the example says, return lists False, not True
    # >>is_iterator([1, 2, 3])
    # >>False
    is_iterator,
)
