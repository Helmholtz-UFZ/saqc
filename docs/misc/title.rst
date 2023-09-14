.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later


===========================================
SaQC - System for automated Quality Control
===========================================

Anomalies and errors are the rule not the exception when working with
time series data. This is especially true, if such data originates
from in-situ measurements of environmental properties.
Almost all applications, however, implicitly rely on data, that complies
with some definition of 'correct'.
In order to infer reliable data products and tools, there is no alternative
to quality control. SaQC provides all the building blocks to comfortably
bridge the gap between 'usually faulty' and 'expected to be corrected' in
a accessible, consistent, objective and reproducible way.


-------------
Documentation
-------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: gettingstarted
      :link-type: ref

      * installation
      * first steps
      * python API introduction
      * command line syntax
      +++
      *Setting up and test-running SaQC on your system*

   .. grid-item-card:: SaQC Configurator
      :link: https://webapp.ufz.de/saqc-config-app/

      * parametrisation tool and sand-box for all the SaQC methods
      * accessible without having any environment
      +++
      *Configuring and testing SaQC in a readily available sandbox*


   .. grid-item-card:: Functionality Overview (API)
      :link: funcs
      :link-type: ref

      * flagging methods overview
      * processing algorithms overview
      * tools overview
      +++
      *Overview of the API access to the available algorithms*


   .. grid-item-card:: Cookbooks
      :link: cookbooks
      :link-type: ref

      * outlier detection
      * frequency alignment
      * drift detection
      * data modelling
      * wrapping generic or custom functionality
      +++
      *Step-by-step guides to the implementation of basic QC tasks in SaQC*


   .. grid-item-card:: Documentation
      :link: ../documentation/documentationPanels
      :link-type: doc

      * CSV file-controlled flagging
      * global keywords
      * customization
      +++
      *Introductions to core mechanics and principles*


   .. grid-item-card:: Developer Resources
      :link: ../devresources/devResPanels
      :link-type: doc

      * writing documentation
      * implementing SaQC methods
      +++
      *All the materials needed, to get involved in SaQC development*


SaQC is developed and maintained by the
`Research Data Management <https://www.ufz.de/index.php?en=45348>`_ Team at the
`Helmholtz-Centre for Environmental Research - UFZ <https://www.ufz.de/>`_.
It manifests the requirements and experiences made from the implementation and
operation of fully automated quality control pipelines for environmental sensor data.
The diversity of communities involved in this process and the special needs within the
realm of scientific data acquisition and its provisioning, have shaped SaQC into
its current state. We define this state as: inherently consistent, yet externally
extensible, traceable, approachable for non-programmers and usable in a wide range
of applications, from exploratory interactive programming environments to large-scale
fully automated, managed workflows.

