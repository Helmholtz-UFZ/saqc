.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Customizations
==============

SaQC comes with a continuously growing number of pre-implemented quality-checking and processing
routines as well as flagging schemes. For a sufficiently large use case, however, it might be
necessary to extend the system anyhow. The main extension options, namely
:ref:`quality check routines <documentation/Customizations:custom quality check routines>`
and the :ref:`flagging scheme <documentation/Customizations:custom flagging schemes>`.
Both of these mechanisms are described within this document.

Custom Quality Check Routines
-----------------------------

In case you are missing quality check routines, you are, of course, very welcome to file a feature request issue on the project's `GitLab repository <https://git.ufz.de/rdm-software/saqc>`_. However, if you are more the "I-get-this-done-by-myself" type of person, SaQC offers the possibility to directly extend its functionality using its interface to the evaluation machinery.

In order to make a function usable within the evaluation framework of SaQC, it needs to implement the following function interface:


.. code-block:: python

   import saqc

   def yourTestFunction(qc: SaQC, field: str | list[str], *args, **kwargs) -> SaQC:
       # your code
       return qc


with the following parameters

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``qc``
     - An instance of ``SaQC``
   * - ``field``
     - The field(s)/column(s) of ``data`` the function is processing/flagging.
   * - ``args``
     - Any number of named arguments needed to parameterize the function.
   * - ``kwargs``
     - Any number of named keyword arguments needed to parameterize the function. ``kwargs``
       need to be present, even if the function needs no keyword arguments at all


Integrate into SaQC
^^^^^^^^^^^^^^^^^^^

SaQC provides two decorators, :py:func:`@flagging` and :py:func:`@register`, to integrate custom functions
into its workflow. The choice between them depends on the nature of your algorithm. :py:func:`@register`
is a more versatile decorator, allowing you to handle masking, demasking, and squeezing of data and flags, while
:py:func:`@flagging` is simpler and suitable for univariate flagging functions without the need for complex
data manipulations.

Use :py:func:`@flagging` for simple univariate flagging tasks without the need for complex data manipulations.
:py:func:`@flagging` is especially suitable when your algorithm operates on a single column


.. code-block:: python

   from saqc import SaQC
   from saqc.core.register import flagging

   @flagging()
   def simpleFlagging(saqc: SaQC, field: str | list[str], param1: ..., param2: ..., **kwargs) -> SaQC:
       """
       Your simple univariate flagging logic goes here.

       Parameters
       ----------
       saqc : SaQC
           The SaQC instance.
       field : str
          The field or fields on which to apply anomaly detection.
       param1 : ...
           Additional parameters needed for your algorithm.
       param2 : ...
           Additional parameters needed for your algorithm.

       Returns
       -------
       SaQC
           The modified SaQC instance.
       """
       # Your flagging logic here
       # Modify saqc._flags as needed
       return saqc


Use :py:func:`@register` when your algorithm needs to handle multiple columns simultaneously (``multivariate=True``)
and or you need explicit control over masking, demasking, and squeezing of data and flags.
:py:func:`register` is especially for complex algorithms that involve interactions between different columns.


.. code-block:: python

   from saqc import SaQC
   from saqc.core.register import register

   @register(
       mask=["field"], # Parameter(s) of the decorated functions giving the names of columns in SaQC._data to mask before the call
       demask=["field"], # Parameter(s) of the decorated functions giving the names of columns in SaQC._data to unmask after the call
       squeeze=["field"], # Parameter(s) of the decorated functions giving the names of columns in SaQC._flags to squeeze into a single flags column after the call
       multivariate=True,  # Set to True to handle multiple columns
       handles_target=False,
   )
   def complexAlgorithm(
       saqc: SaQC, field: str | list[str], param1: ..., param2: ..., **kwargs
   ) -> SaQC:
       """
       Your custom anomaly detection logic goes here.

       Parameters
       ----------
       saqc : SaQC
           The SaQC instance.
       field : str or list of str
           The field or fields on which to apply anomaly detection.
       param1 : ...
           Additional parameters needed for your algorithm.
       param2 : ...
           Additional parameters needed for your algorithm.

       Returns
       -------
       SaQC
           The modified SaQC instance.
       """
       # Your anomaly detection logic here
       # Modify saqc._flags and saqc._data as needed
       return saqc



Custom flagging schemes
-----------------------

Sorry for the inconvenience! Coming soon...
