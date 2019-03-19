#! /usr/bin/env python
# -*- coding: utf-8 -*-

from funcs import funcs
import numpy as np


class Fields:
    VARNAME = "headerout"
    STARTDATE = "date start"
    ENDDATE = "date end"
    FLAGS = "Flag*"


class Params:
    NAME = "name"
    FUNC = "func"
    FLAGPERIOD = "flag_period"
    FLAGVALUES = "flag_values"
    FLAG = "flag"


FUNCMAP = {
    "maintenance": funcs.flagMaintenance,
    "man_flag": funcs.flagManual,
    "MAD": funcs.flagMad,
    "constant": funcs.flagConstant
}

NODATA = np.nan
