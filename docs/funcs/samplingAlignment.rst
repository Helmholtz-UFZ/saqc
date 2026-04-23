.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Resampling
----------

Methods for transforming time series data through resampling and alignment, including
custom aggregation, alignment to a regular frequency grid, and back projection of
flags to the original data.

.. currentmodule:: saqc

.. autosummary::
   :nosignatures:

   ~SaQC.align
   ~SaQC.concatFlags
   ~SaQC.resample
   ~SaQC.reindex
