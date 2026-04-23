.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Data Products
-------------

Methods for deriving additional data products from time series data, such as smoothed
signals using frequency filters or polynomial fits, residual extraction, and calculation
of kNN and LOF scores.

.. currentmodule:: saqc

.. autosummary::
   :nosignatures:

   ~SaQC.assignKNNScore
   ~SaQC.assignLOF
   ~SaQC.assignUniLOF
   ~SaQC.assignZScore
   ~SaQC.calculatePolynomialResiduals
   ~SaQC.calculateRollingResiduals
   ~SaQC.fitPolynomial
   ~SaQC.fitLowpassFilter
