.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _global_function_parameters:

Global Function Parameters
==========================

SaQC provides a number of global function parameters that are available
in every SaQC function. These parameters modify the *execution context*
of a function rather than its algorithmic behavior, which is controlled
by function-specific parameters.

The following global parameters are available.

.. _global_aliasing:

Aliasing
--------

``label``
    The ``label`` keyword allows assigning a string alias to the execution
    of a SaQC function. This alias is used in subsequent calls to
    :py:meth:`saqc.SaQC.plot` and can also be accessed within
    :ref:`custom flagging schemes <custom_flagging_schemes>`.

.. _global_flagging:

Flagging
--------

``flag``
    By default, all SaQC functions mark detected anomalies using the
    :ref:`anomaly flag <internal_flag_anomaly>` and its
    :ref:`flagging scheme <flagging_schemes>` specific
    :ref:`external representation <external_flags>`. This behavior can be
    modified by passing a ``flag`` argument. Detected anomalies will then
    be marked with the given value.

    The provided value must be part of the active
    :ref:`flagging scheme <flagging_schemes>`.

``dfilter``
    Overrides the default filtering threshold defined by the active
    :ref:`flagging scheme <flagging_schemes>`. For details on the
    filtering mechanism, see :ref:`filtering`.

The data type of both arguments depends on the chosen
:ref:`flagging scheme <flagging_schemes>`.

- For :py:class:`~saqc.core.FloatScheme` and
  :py:class:`~saqc.core.AnnotatedFloatScheme`, a floating-point value
  must be provided.
- For :py:class:`~saqc.core.SimpleScheme`, only the literals
  ``"UNFLAGGED"``, ``"OK"``, and ``"BAD"`` are valid.

.. _global_temporal:

Temporal Specification
----------------------

``start_date``
    Extends the flag-related :ref:`masking mechanism <filtering>` by a
    temporal component. Only observations with timestamps greater than or
    equal to ``start_date`` are passed to the function.

``end_date``
    Extends the flag-related :ref:`masking mechanism <filtering>` by a
    temporal component. Only observations with timestamps less than or
    equal to ``end_date`` are passed to the function.

Both arguments may be provided as strings, ``pandas.Timestamp``,
or ``datetime.datetime`` objects.

While the latter two are interpreted exactly as given, string
representations allow partial datetime specifications to restrict the
temporal context of function execution.

Examples:

- ``start_date="01:00"`` and ``end_date="04:00"``
  Only observations between the first and fourth minute of every hour
  are processed.

- ``start_date="15:00:00"`` and ``end_date="17:00:00"``
  Only observations between 15:00 and 17:00 of every day are processed.

- ``start_date="01T15:00:00"`` and ``end_date="13T17:30:00"``
  Only observations between the first day at 15:00 and the 13th day at
  17:30 of every month are processed.

- ``start_date="01-01T00:00:00"`` and
  ``end_date="02-28T23:59:59"``
  Only observations from January and February of every year are processed.

.. _global_examples:

Examples
--------

.. _example_flagging_constraint:

Flagging Scheme Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following examples assume the default
:ref:`flagging scheme <flagging_schemes>`,
which is :py:class:`~saqc.core.FloatScheme`.

.. _example_data:

Example Data
~~~~~~~~~~~~

.. plot::
   :context: close-figs
   :include-source: False

   import pandas as pd
   import numpy as np
   from saqc import SaQC

   noise = np.random.normal(0, 1, 200)
   data = pd.Series(noise, index=pd.date_range('2020','2021',periods=200), name='data')
   data.iloc[20] = 16
   data.iloc[100] = -17
   data.iloc[160:180] = -3

   qc = SaQC(data)

Let us generate some example data and plot it:

.. doctest:: globalParameters

   >>> import pandas as pd
   >>> import numpy as np
   >>> from saqc import SaQC
   >>> noise = np.random.normal(0, 1, 200) # some normally distributed noise
   >>> data = pd.Series(noise, index=pd.date_range('2020','2021',periods=200), name='data') # index the noise with some dates
   >>> data.iloc[20] = 16 # add some artificial anomalies:
   >>> data.iloc[100] = -17
   >>> data.iloc[160:180] = -3
   >>> qc = SaQC(data)
   >>> qc.plot('data')  # doctest:+SKIP

.. plot::
   :context: close-figs
   :include-source: False

   qc.plot('data')

.. _example_label_keyword:

Label Keyword
~~~~~~~~~~~~~

The ``label`` argument assigns a custom identifier to a SaQC function
call. This directly affects subsequent calls to
:py:meth:`saqc.SaQC.plot`.

It is especially useful for enriching figures with contextual
information and for distinguishing results from different function
calls.

.. doctest:: globalParameters

   >>> qc = SaQC(data)
   >>> qc = qc.flagRange('data', max=15, label='values < 15')
   >>> qc = qc.flagRange('data', min=-16, label='values > -16')
   >>> qc.plot('data')  # doctest:+SKIP

.. _example_dfilter_flag:

``dfilter`` and ``flag``
~~~~~~~~~~~~~~~~~~~~~~~~

To illustrate the interplay of ``flag`` and ``dfilter``,
we first assign custom flag levels:

.. doctest:: globalParameters

   >>> qc = SaQC(data)
   >>> qc = qc.flagRange('data', max=15, label='flaglevel=200', flag=200)
   >>> qc = qc.flagRange('data', min=-16, label='flaglevel=100', flag=100)
   >>> qc.plot('data')  # doctest:+SKIP

Using the ``dfilter`` keyword, we can control which observations are
passed to a function. For example:

.. doctest::

   >>> qc.plot('data', dfilter=50)  # doctest:+SKIP

.. _example_flag_separation:

Flag Separation
^^^^^^^^^^^^^^^

Usually ``dfilter`` equals to the anomaly flag of the active
:ref:`flagging scheme <flagging_schemes>`.

If a function assigns this same flag value, subsequent calls will not
process already flagged observations.

.. doctest:: globalParameters

   >>> qc = SaQC(data)
   >>> qc = qc.flagRange('data', max=15, label='value > 15')
   >>> qc = qc.flagRange('data', max=0, label='value > 0')
   >>> qc.plot('data')  # doctest:+SKIP

To re-test already flagged observations, increase or disable the
filtering threshold:

.. doctest:: globalParameters

   >>> from saqc.constants import FILTER_NONE
   >>> qc = qc.flagRange('data', max=0, label='value > 0', dfilter=FILTER_NONE)
   >>> qc.plot('data')  # doctest:+SKIP

.. _example_unflagging:

Unflagging
^^^^^^^^^^

The ``flag`` keyword can also be used to remove flags from observations.

For :py:class:`~saqc.core.FloatScheme`, the
:ref:`internal unchecked flag <internal_flag_unflagged>` is
``-np.inf``. Alternatively, the constant
:py:attr:`~saqc.constants.UNFLAGGED` may be used.

To override existing flags, the input filter must be raised or disabled:

.. doctest:: globalParameters

   >>> from saqc.constants import UNFLAGGED, FILTER_NONE
   >>> qc = qc.flagConstants(
   ...     'data',
   ...     window='2D',
   ...     thresh=0,
   ...     dfilter=FILTER_NONE,
   ...     flag=UNFLAGGED
   ... )
   >>> qc.plot('data')  # doctest:+SKIP
