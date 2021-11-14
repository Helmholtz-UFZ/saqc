
.. role:: raw-html(raw)
    :format: html

.. |ufzLogo| image:: /ressources/images/Representative/UFZ_Logo.jpg
    :width: 45 %
    :target: https://www.ufz.de/


.. |rdmLogo| image:: /ressources/images/Representative/RDMlogo.jpg
    :width: 30 %
    :target: https://www.ufz.de/index.php?de=45348


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


.. image:: /ressources/images/Representative/gitlabLogo.png
    :width: 17 %
    :align: right
    :target: https://git.ufz.de/rdm-software/saqc


|ufzLogo| |rdmLogo|

===========================================
SaQC - System for automated Quality Control
===========================================


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
      - * get SaQC from PyPI
        * use SaQC as a commandline application and configure your flagging pipelines via plain .csv files
        * or use the SaQC python API, enabling integration in your python processing script
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
      - * modifyyour data, by interpolations, corrections and transformations
        * calculate data products, such as residues or outlier scores
        * automatically keep track of labeling history and label significance
    * - |sacMV|
      - * apply multivariate flagging function


