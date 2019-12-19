#! /usr/bin/env python
# -*- coding: utf-8 -*-

from saqc.flagger.categoricalflagger import CategoricalFlagger


FLAGS = [-1, 0, 1]


class SimpleFlagger(CategoricalFlagger):
    def __init__(self):
        super().__init__(FLAGS)
