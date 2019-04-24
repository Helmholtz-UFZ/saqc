#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import funcs
import numpy as np


class Fields:
    VARNAME = "headerout"
    STARTDATE = "date start"
    ENDDATE = "date end"
    ASSIGN = "assign"
    FLAGS = "Flag*"


class Params:
    FUNC = "func"
    FLAGPERIOD = "flag_period"
    FLAGVALUES = "flag_values"
    FLAG = "flag"


# FUNCMAP = {
#     "manflag": funcs.flagManual,
#     "mad": funcs.flagMad,
#     "constant": funcs.flagConstant,
#     "generic": funcs.flagGeneric
# }
