.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _funcs:


Functionality Overview
----------------------
..
   Anomaly Detection
   ------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: Anomaly Detection

      .. grid:: 2
         :gutter: 2

         .. grid-item-card:: Basic Anomaly Detection
            :link: basicAnomalies
            :link-type: doc

            * data *gaps*,
            * data *jumps*,
            * *isolated* points,
            * *constant* and low variance regimes.
            +++

         .. grid-item-card:: Outlier Detection
            :link: outlierDetection
            :link-type: doc

            * rolling *Z-score* cutoff
            * modified local outlier factor (univariate-*LOF*)
            * deterministic *offset pattern* search
            +++

         .. grid-item-card:: Multivariate Analysis
            :link: multivariateAnalysis
            :link-type: doc

            * k-nearest neighbor scores (*kNN*)
            * local outlier factor (*LOF*)
            +++

         .. grid-item-card:: Distributional Analysis
            :link: distributionalAnomalies
            :link-type: doc

            * detect *change points*
            * detect continuous *noisy* data sections
            +++


   .. grid-item-card:: Data and Flag Tools

      .. grid:: 2
         :gutter: 2

         .. grid-item-card:: Data Independent Flags Manipulation
            :link: flagTools
            :link-type: doc

            * *copy* flags
            * *transfer* flags
            * *propagate* flags
            * *force*-set unitary or precalculated flags values
            +++

         .. grid-item-card:: Basic tools
            :link: tools
            :link-type: doc

            * plot variables
            * copy and delete variables
            +++

         .. grid-item-card:: Generic and Custom Functions
            :link: genericWrapper
            :link-type: doc

            * basic *logical* aggregation of variables
            * basic *arithmetical* aggregation of variables
            * *custom functions*
            * *rolling*, *resampling*, *transformation*
            +++

   .. grid-item-card:: Data Manipulation

      .. grid:: 2
         :gutter: 2

         .. grid-item-card:: Data Products
            :link: dataProducts
            :link-type: doc

            * smooth with *frequency filter*
            * smooth with *polynomials*
            * obtain *residuals* from smoothing
            * obtain *kNN* or *LOF* scores
            +++

         .. grid-item-card:: Resampling
            :link: samplingAlignment
            :link-type: doc

            * *resample* data using custom aggregation
            * *align* data to frequency grid with minimal data distortion
            * *back project* flags from aligned data onto original series
            +++

   .. grid-item-card:: Data Correction

      .. grid:: 2
         :gutter: 2

         .. grid-item-card:: Gap filling
            :link: filling
            :link-type: doc

            * fill gaps with *interpolations*
            * fill gaps using a *rolling* window
            +++

         .. grid-item-card:: Drift Detection and Correction
            :link: driftBehavior
            :link-type: doc

            * deviation predicted by a *model*
            * deviation from the *majority* of parallel curves
            * deviation from a defined *norm* curve
            +++
