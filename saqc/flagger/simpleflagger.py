#! /usr/bin/env python
# -*- coding: utf-8 -*-


from .baseflagger import BaseFlagger


FLAGS = [-1, 0, 1]


class SimpleFlagger(BaseFlagger):

    def __init__(self):
        super().__init__(FLAGS)
