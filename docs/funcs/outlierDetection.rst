.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Outlier Detection
-----------------

Methods for detecting outliers in time series data, including rolling Z-score thresholds,
modified univariate Local Outlier Factor (LOF), and deterministic offset pattern detection.

.. currentmodule:: saqc

.. autosummary::
   :nosignatures:

   ~SaQC.flagUniLOF
   ~SaQC.flagByStray
   ~SaQC.flagOffset
   ~SaQC.flagRange
   ~SaQC.flagLOF
   ~SaQC.flagZScore
   ~SaQC.flagPlateau

