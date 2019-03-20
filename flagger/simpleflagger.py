#! /usr/bin/env python
# -*- coding: utf-8 -*-

from .baseflagger import BaseFlagger


class SimpleFlagger(BaseFlagger):
    def __init__(self):
        super().__init__(0, 1)
