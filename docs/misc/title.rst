.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. |sacRaw| image:: /resources/images/representative/RawData.png
    :height: 150 px
    :width: 288 px

.. |sacFlagged| image:: /resources/images/representative/DifferentFlags.png
    :height: 150 px
    :width: 288 px

.. |sacMV| image:: /resources/images/representative/MultivarSAC.png
    :height: 150 px
    :width: 288 px

.. |sacProc| image:: /resources/images/representative/ProcessingDrift.png
    :height: 150 px
    :width: 288 px

.. |pyLogo| image:: /resources/images/representative/PythonLogo.png
    :height: 108 px
    :width: 105 px

.. |csvConfig| image:: /resources/images/representative/CsvConfig.png
    :height: 100 px
    :width: 176 px

.. |legendEXMPL| image:: /resources/images/representative/LegendEXMPL.png
    :height: 100 px
    :width: 200


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

--------
Features
--------

.. list-table::

    * - |pyLogo| |csvConfig|
      - * :ref:`get and install SaQC <gettingstarted/InstallationGuide:installation guide>`
        * :ref:`use the SaQC python API, enabling integration into larger programs <gettingstarted/TutorialAPI:python api>`
        * or use SaQC as a commandline application and configure your pipelines via plain text
    * - |sacRaw|
      - * easily load data from multiple sources, concatenating them in a SaQC object
        * :ref:`preprocess your data, by aligning it to shared frequency grids <cookbooks/DataRegularisation:Data Regularization>`
    * - |sacFlagged|
      - * apply basic plausibility checks, as well as
        * more complex, univariat flagging Functions
    * - |legendEXMPL|
      - * automatically keep track of flagging history and flags significance for every datapoint
        * define and use custom schemes to translate your flags to and from SaQC
    * - |sacProc|
      - * modify your data by :ref:`interpolations <cookbooks/DataRegularisation:Interpolation>`, corrections and :ref:`transformations <cookbooks/DataRegularisation:Aggregation>`
        * calculate data products, such as :ref:`residuals or outlier scores <cookbooks/OutlierDetection:Residuals and Scores>`
    * - |sacMV|
      - * apply :ref:`multivariate flagging functions <cookbooks/MultivariateFlagging:Multivariate Flagging>`
