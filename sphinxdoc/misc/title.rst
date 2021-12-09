
.. |sacRaw| image:: /ressources/images/Representative/RawData.png
    :height: 150 px
    :width: 288 px

.. |sacFlagged| image:: /ressources/images/Representative/DifferentFlags.png
    :height: 150 px
    :width: 288 px

.. |sacMV| image:: /ressources/images/Representative/MultivarSAC.png
    :height: 150 px
    :width: 288 px

.. |sacProc| image:: /ressources/images/Representative/ProcessingDrift.png
    :height: 150 px
    :width: 288 px

.. |pyLogo| image:: /ressources/images/Representative/pythonLogo.png
    :height: 108 px
    :width: 105 px

.. |csvConfig| image:: /ressources/images/Representative/csvConfig.png
    :height: 100 px
    :width: 176 px


.. |legendEXMPL| image:: /ressources/images/Representative/legendEXMPL.png
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
      - * :ref:`get and install SaQC <getting_started/InstallationGuide:installation guide>`
        * :ref:`use the SaQC python API, enabling integration into larger programs <getting_started/TutorialAPI:python api>`
        * or use SaQC as a commandline application and configure your pipelines via plain text
    * - |sacRaw|
      - * easily load data from multiple sources, concatenating them in a SaQC object
        * :ref:`preprocess your data, by aligning it to shared frequency grids <cook_books/DataRegularisation:Data Regularisation>`
    * - |sacFlagged|
      - * apply basic plausibility checks, as well as
        * more complex, univariat flagging Functions
    * - |legendEXMPL|
      - * automatically keep track of flagging history and flags significance for every datapoint
        * define and use custom schemes to translate your flags to and from SaQC
    * - |sacProc|
      - * modify your data by :ref:`interpolations <cook_books/DataRegularisation:Interpolation>`, corrections and :ref:`transformations <cook_books/DataRegularisation:Aggregation>`
        * calculate data products, such as :ref:`residues or outlier scores <cook_books/OutlierDetection:Residues and Scores>`
    * - |sacMV|
      - * apply :ref:`multivariate flagging functions <cook_books/MultivariateFlagging:Multivariate Flagging>`
