
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

SaQC is developed and maintained by the
`Research Data Management <https://www.ufz.de/index.php?en=45348>`_ Team at the
`Helmholz-Centre for Environmental Research - UFZ <https://www.ufz.de/>`_.
It manifests the requirements and experiences made from establishment and operation of
fully automated quality control pipelines for environmental sensor data. 
The diversity of scientific communities involved and the special needs within the
realm of scientific data aqcuisition and its provisioning have shaped SaQC into
its current state.

We define SaQC: inherently consistent, yet externally extensible, traceable,
approachable for non-programmers and usable in a wide range of applications, from
exploratory interactive programming environments to large-scale fully automated,
managed workflows.

..
   The number of involved scientific communities is large, ranging from hydrology to
   climate sciences


   obtained from scientific communities like water, soil and climate sciences.

   SaQC by the :ref:`Research Data Management<https://www.ufz.de/index.php?de=45348>`_
   Team at the :ref:`Helmholz-Centre for Environmental Research - UFZ<https://www.ufz.de/>`_
   It builds

   SaQC aims to be
   - consitent
   - extesible
   - 

   Quality Control of numerical data requires a significant amount of
   domain knowledge and practical experience. Finding a robust setup of
   quality tests that identifies as many suspicious values as possible, without
   removing valid data, is usually a time-consuming endeavor,
   even for experts. SaQC is both, a Python framework and a command line application, that
   addresses the exploratory nature of quality control by offering a
   continuously growing number of quality check routines through a flexible
   and simple configuration system.


   Below its user interface, SaQC is highly customizable and extensible.
   A modular structure and well-defined interfaces make it easy to extend
   the system with custom quality checks. Furthermore, even core components like
   the flagging scheme are exchangeable.

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
        * preprocess your data, by aligning it to shared frequency grids
    * - |sacFlagged|
      - * apply basic plausibility checks, as well as
        * more complex, univariat flagging Functions
    * - |legendEXMPL|
      - * automatically keep track of flagging history and flags significance for every datapoint
        * define and use custom schemes to translate your flags to and from SaQC
    * - |sacProc|
      - * modify your data by interpolations, corrections and transformations
        * calculate data products, such as residues or outlier scores
    * - |sacMV|
      - * apply multivariate flagging function
