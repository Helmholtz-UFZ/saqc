.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Customizations
==============

SaQC comes with a continuously growing number of pre-implemented
quality checking and processing routines as well as flagging schemes. 
For any sufficiently large use case however, it is very likely that the 
functions provided won't fulfill all your needs and requirements.

Acknowledging the impossibility to address all imaginable use cases, we 
designed the system to allow for extensions and costumizations. The main extensions options, namely 
:ref:`quality check routines <documentation/Customizations:custom quality check routines>`
and the :ref:`flagging scheme <documentation/Customizations:custom flagging schemes>`
are described within this documents.

Custom quality check routines
-----------------------------

In case you are missing quality check routines, you are of course very
welcome to file a feature request issue on the project's
`gitlab repository <https://git.ufz.de/rdm-software/saqc>`_. However, if 
you are more the "I-get-this-done-by-myself" type of person,
SaQC provides two ways to integrate custom routines into the system:


#. The :ref:`extension language <documentation/GenericFunctions:Generic Functions>`
#. An :ref:`interface <documentation/Customizations:interface>` to the evaluation machinery

Interface
^^^^^^^^^

In order to make a function usable within the evaluation framework of SaQC, it needs to
implement the following function interface

.. code-block:: python

   import pandas
   import dios
   import saqc

   def yourTestFunction(
      saqc: SaQC
      field: str,
      *args,
      **kwargs
      ) -> SaQC

Argument Descriptions
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``data``
     - The actual dataset, an instance of ``dios.DictOfSeries``.
   * - ``field``
     - The field/column within ``data``, that function is processing.
   * - ``flags``
     - An instance of saqc.Flags, responsible for the translation of test results into quality attributes.
   * - ``args``
     - Any other arguments needed to parameterize the function.
   * - ``kwargs``
     - Any other keyword arguments needed to parameterize the function.


Integrate into SaQC
^^^^^^^^^^^^^^^^^^^

In order make your function available to the system it needs to be registered. We provide a decorator 
`\ ``flagging`` <saqc/functions/register.py>`_ with saqc, to integrate your 
test functions into SaQC. Here is a complete dummy example:

.. code-block:: python

   from saqc import register

   @flagging()
   def yourTestFunction(saqc: SaQC, field: str, *args, **kwargs):
       return saqc

Example
^^^^^^^

The function `\ ``flagRange`` <saqc/funcs/outliers.py>`_ provides a simple, yet complete implementation of 
a quality check routine. You might want to look into its implementation as an example.

Custom flagging schemes
-----------------------

Sorry for the inconvenience! Coming soon...
