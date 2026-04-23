.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Drift Detection and Correction
------------------------------

Methods for detecting and correcting drift in time series data, including
deviations from model predictions, the majority of parallel series, and
predefined reference curves.

.. currentmodule:: saqc

.. autosummary::
   :nosignatures:

   ~SaQC.flagDriftFromNorm
   ~SaQC.flagDriftFromReference
   ~SaQC.correctDrift
   ~SaQC.correctRegimeAnomaly
   ~SaQC.correctOffset
   ~SaQC.flagRegimeAnomaly
   ~SaQC.assignRegimeAnomaly
