#! /usr/bin/env python
# -*- coding: utf-8 -*-


from saqc.flagger.categoricalflagger import CategoricalBaseFlagger


FLAGS = [-1, 0, 1]


class SimpleFlagger(CategoricalBaseFlagger):
    def __init__(self):
        super().__init__(FLAGS)
