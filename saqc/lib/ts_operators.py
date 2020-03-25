#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

OP_MODULES = {'pd': pd,
              'np': np
              }


def eval_func_string(func_string):
    module_dot = func_string.find(".")
    if module_dot > 0:
        module = func_string[:module_dot]
        if module in OP_MODULES:
            op_dot = func_string.rfind(".")
            return getattr(OP_MODULES[module], func_string[op_dot+1:])
        else:
            availability_list = ['"' + k + '"' +  " (= " + str(s.__name__) + ")" for k,s in (OP_MODULES.items())]
            availability_list = " \n".join(availability_list)
            raise ValueError('The external-module alias "{}" is not known to the timeseries functions dispatcher.'
                             '\n'
                             'Please select from:'
                             '\n'
                             '{}'
                             '\n'.format(module, availability_list))


def compose_function(functions):
    def composed(ts):
        for func in reversed(functions):
            ts = func(ts)
        return ts
    return composed




# ts_transformations
def diff_log(ts):
    return np.log(ts / ts.shift(1))


def deri_log(ts, unit='1min'):
    return diff_log(ts)/delta_t(ts, unit=unit)


def delta_t(ts, unit='1min'):
    return ts.index.to_series().diff().dt.total_seconds() / pd.Timedelta(unit).total_seconds()