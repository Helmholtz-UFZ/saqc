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



.. _custom_flagging_schemes:

Custom Flagging Schemes
~~~~~~~~~~~~~~~~~~~~~~~

SaQC provides several built-in :ref:`flagging schemes <flagging_schemes>`
and supports the implementation of custom schemes via a well-defined
extension interface.

Custom flagging schemes must subclass
:py:class:`saqc.core.translation.TranslationScheme` and implement the
following abstract interface:

.. code-block:: python

    class TranslationScheme:
        @property
        @abstractmethod
        def DFILTER_DEFAULT(self) -> float:
            pass

        @abstractmethod
        def __call__(self, flag: EXTERNAL_FLAG) -> float:
            pass

        @abstractmethod
        def toInternal(self, flags: pd.DataFrame | DictOfSeries) -> Flags:
            """
            Translate from external flags to internal flags.
            """
            pass

        @abstractmethod
        def toExternal(
            self,
            flags: Flags,
            attrs: dict | None = None
        ) -> DictOfSeries:
            """
            Translate from internal flags to external flags.
            """
            pass

``DFILTER_DEFAULT`` defines the default filtering constant of the respective
flagging scheme (see :ref:`section filtering <filtering>`). The methods ``toInternal`` and
``toExternal`` implement the translation between external and internal flags.
The ``__call__`` method translates a single external flag into its internal
representation.
In addition to these structural requirements, there is also a semantic
prerequisite: every flagging scheme must provide a direct translation for
the two relevant :ref:`internal flags <internal_flags>` ``-numpy.inf`` and
``255.0`` (the anomaly marker).

For simple flagging schemes that directly map scalar flag values to one
another, the base class
:py:class:`saqc.core.translation.MappingScheme` may provide a more convenient
implementation.

The implementation of :py:class:`saqc.core.translation.SimpleScheme` may serve as
an illustrative example:

.. code-block:: python

    import numpy as np
    from saqc.core.translation import MappingScheme

    class SimpleScheme(MappingScheme):
        """
        Acts as the default translator and provides a changeable subset
        of the internal float flags.
        """

        _FORWARD = {
            "UNFLAGGED": -np.inf,
            "BAD": 255.0,
            "OK": 0,
        }

        _BACKWARD = {
            -np.inf: "UNFLAGGED",
            np.nan: "UNFLAGGED",
            255.0: "BAD",
            0: "OK",
        }

        def __init__(self):
            super().__init__(forward=self._FORWARD, backward=self._BACKWARD)


