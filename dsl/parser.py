#! /usr/bin/env python
# -*- coding: utf-8 -*-

import yaml


def parseFlag(expr):
    try:
        # CLoader needs the debian package libyaml-dev, if it is
        # not present fall back on default loader.
        from yaml import CSafeLoader as SafeLoader
    except ImportError:
        from yaml import SafeLoader as SafeLoader
    content = yaml.load("[{:}]".format(expr), Loader=SafeLoader)
    name = content[0]
    out = {}
    for pdict in content[1:]:
        out.update(pdict)
    return name, out
