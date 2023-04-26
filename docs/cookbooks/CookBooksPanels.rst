.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _cookbooks:

Cook Books
==========

.. toctree::
   :caption: Cookbooks
   :maxdepth: 1
   :hidden:

   DataRegularisation
   OutlierDetection
   ResidualOutlierDetection
   MultivariateFlagging
   ../documentation/GenericFunctions


.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Data Alignment
      :link: DataRegularisation
      :link-type: doc

      * modify data, so that its index exhibits a unary sampling rate
      * by: shifting, interpolating or aggregating it
      * but: conserve data gap structure
      * minimize and control value distortion from alignment
      * back-project calculated flags from the aligned data onto the original
      +++
      *Obtain representative data derivative sampled at an evenly spaced frequency grid*


   .. grid-item-card:: Outlier Detection
      :link: OutlierDetection
      :link-type: doc

      * quickly set up a simple yet powerful outlier detection algorithm
      * learn to interprete and tune the parameters
      +++
      *Introduction to the Univariat Local Outlier Factor Algorithm*

   .. grid-item-card:: Multivariate Outlier Detection
      :link: MultivariateFlagging
      :link-type: doc

      * apply k-nearest neighbor scoring to obtain outlier evaluation in multivariate contexts
      * use STRAY Algorithm to find a suitable cut-off point for obtained scores
      +++
      *Scoring data in multivariate context*


   .. grid-item-card:: Generic Expressions and Custom Functionality
      :link: ../documentation/GenericFunctions
      :link-type: doc

      * obtain results from arbitrary arithmetic operations on your data
      * freely formulate logical quality control conditions
      +++
      *Wrap your custom logical and arithmetic expressions with the generic functions*

   .. grid-item-card:: Modelling, Residuals and Arithmetics
      :link: ResidualOutlierDetection
      :link-type: doc

      * obtain data derivates through different modelling approaches
      * like rolling statistics or curve fits
      * obtain model errors and apply standard anomaly tests on those
      * project the result onto the original data
      +++
      *How to derive flagging assertions from error models*
