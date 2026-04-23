.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later


===========================================
SaQC - System for automated Quality Control
===========================================

SaQC is an open-source framework for automated, transparent, and reproducible
quality control of time series data. It transforms raw time series data into
trustworthy data products by making quality control an explicit step in
FAIR-compliant workflows, enabling reliable use in applications such as
monitoring, modelling, and decision-making.

Quality control logic in SaQC can be defined using its Python API or through
structured, low-code configuration files. The low-code approach enables
domain experts to define checks, compound flagging strategies, and processing
steps with minimal programming effort—and to apply the same rules consistently
to both historical archives and live data streams.

A distinctive feature of SaQC is its flexible quality annotation, which provides
a complete, observation-level flag history to ensure end-to-end provenance,
traceability, and auditability. Its anomaly detection capabilities range from
classical validation methods to advanced techniques. Most components of SaQC,
including quality annotation and QC functionality, are easily extensible through
well-defined interfaces, enabling hybrid rule-based and machine learning workflows.


.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: gettingstarted
      :link-type: ref

      * installation and setup
      * first steps
      * Python API introduction
      * command-line usage
      +++
      *Set up and run SaQC on your system*

   .. grid-item-card:: Functionality Overview (API)
      :link: ../modules/SaQCFunctions
      :link-type: doc

      * overview of flagging methods
      * overview of processing algorithms
      * overview of tools
      +++
      *Explore the API and available algorithms*

   .. grid-item-card:: Documentation
      :link: ../documentation/documentationPanels
      :link-type: doc

      * configuration-based quality control
      * global keywords
      * flags and flagging
      * customization
      +++
      *Understand core concepts and system behavior*

   .. grid-item-card:: Cookbooks
      :link: cookbooks
      :link-type: ref

      * outlier detection
      * frequency alignment
      * drift detection
      * data modeling
      * custom and generic function usage
      +++
      *Step-by-step guides for common QC tasks*

   .. grid-item-card:: Galaxy Tool
      :link: https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fufz%2Fsaqc%2Fsaqc%2F2.7.0%2Bgalaxy0&version=latest

      * configure and run SaQC
      * integrate into larger workflows
      +++
      *Run and integrate SaQC within analysis pipelines*

   .. grid-item-card:: Community
      :link: ../community/Community
      :link-type: doc

      * publications
      * users and partners
      +++
      *Explore the SaQC community and ecosystem*


SaQC turns quality control into an explicit, traceable, and version-controlled
step in time series data workflows, enabling the production of AI-ready data
and supporting reliable downstream use in research data portals, environmental
models, and digital twins.

Beyond stand-alone use, SaQC is designed as a modular building block that can
be integrated into various applications. It is, for example, an integral part
of `Neptoon <https://www.neptoon.org/en/latest/>`_, is integrated into
`time.IO <https://codebase.helmholtz.cloud/ufz-tsm>`_ - a time series data
infrastructure developed at the UFZ - and is also available on
`Galaxy Europe <https://usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fufz%2Fsaqc%2Fsaqc%2F2.7.0%2Bgalaxy0&version=latest>`_
for workflow-based, low-barrier execution within larger analysis pipelines.

SaQC is developed and maintained by the
`Research Data Management Team at UFZ <https://www.ufz.de/index.php?en=45348>`_
at the
`Helmholtz Centre for Environmental Research - UFZ <https://www.ufz.de/>`_.
It reflects the requirements and experience gained from implementing and
operating fully automated quality control pipelines for environmental sensor data.

The diversity of involved communities, along with the specific demands of
scientific data acquisition and provisioning, has shaped SaQC into its current
form: inherently consistent yet externally extensible, fully traceable,
accessible to non-programmers, and applicable across a wide range of use cases—
from exploratory, interactive programming environments to large-scale,
fully automated workflows.
