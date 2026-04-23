.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Basic Anomalies
---------------

Detection of fundamental anomalies in time series data, including gaps, sudden jumps,
constant segments, and isolated values.

.. currentmodule:: saqc

.. autosummary::
   :nosignatures:

   ~SaQC.flagNAN
   ~SaQC.flagIsolated
   ~SaQC.flagJumps
   ~SaQC.flagConstants
   ~SaQC.flagByVariance
   ~SaQC.flagPlateau


