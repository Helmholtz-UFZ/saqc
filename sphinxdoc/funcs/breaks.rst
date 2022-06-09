.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

breaks
======

Detecting breaks in data.

This module provides functions to detect and flag breaks in data, for example temporal
gaps (:py:func:`flagMissing`), jumps and drops (:py:func:`flagJumps`) or temporal
isolated values (:py:func:`flagIsolated`).

.. currentmodule:: saqc

.. autosummary::

   ~SaQC.flagMissing
   ~SaQC.flagIsolated
   ~SaQC.flagJumps
