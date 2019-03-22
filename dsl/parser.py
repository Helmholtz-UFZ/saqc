#! /usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

def parseFlag(expr):
    content = yaml.load("[{:}]".format(expr), Loader=yaml.CLoader)
    name = content[0]
    out = {}
    for pdict in content[1:]:
        out.update(pdict)
    return name, out
