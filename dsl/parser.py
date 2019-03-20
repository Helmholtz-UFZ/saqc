#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re


def parseFlag(params):
    out = {}
    for i, part in enumerate(re.split(r";\s*", params)):
        if ":" in part:
            k, v = [i.strip() for i in part.split(":")]
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        elif i == 0:
            out["name"] = part
    return out
