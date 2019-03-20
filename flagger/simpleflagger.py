#! /usr/bin/env python
# -*- coding: utf-8 -*-

from .abstractflagger import AbstractFlagger


class SimpleFlagger(AbstractFlagger):
    def __init__(self):
        super().__init__(0, 1)
