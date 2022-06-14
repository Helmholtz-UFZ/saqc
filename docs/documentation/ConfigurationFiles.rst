.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Configuration Files
===================

In addition to it's :doc:`Python API <../gettingstarted/TutorialAPI>`, SaQC also offers a text based
configuration system. This option has proven as a valuable strategy in largely automated
quality control workflows. The ability to change existing setups without changes
to the source code of, e.g. Extract-Transform-Load (ETL) pipelines, simplies the operation and adaption
of such workflows as well as the collaboration with less technical project partners.

Format
------

Configuration files are expected to be semicolon-separated text files with exactly one header
line. Each row of the configuration file lists one variable and a respective test function that
is applied to the given variable.

Header names
^^^^^^^^^^^^

The first line of every configuration file is dropped, so feel free to use header
names to your liking.


Test function notation
^^^^^^^^^^^^^^^^^^^^^^

The notation of test functions follows the function call notation of Python and
many other programming languages and looks like this:

.. code-block::

   flagRange(min=0, max=100)

Here the function :py:meth:`flagRange <saqc.SaQC.flagRange>` is called and the
values ``0`` and ``100`` are passed to the parameters ``min`` and ``max`` respectively.
As we value readablity of the configuration more than conciseness of the extension language, only
keyword arguments are supported. That means that the notation ``flagRange(0, 100)``
is not a valid replacement for the above example.

Examples
--------

Every row lists one test per variable. If you want to call multiple tests on
a specific variable (and you probably want to), list them in separate rows:

.. code-block::

   varname ; test
   #-------;----------------------------------
   x       ; flagMissing()
   x       ; flagRange(min=0, max=100)
   x       ; flagConstants(window="3h")
   y       ; flagRange(min=-10, max=40)


Available Test Functions
^^^^^^^^^^^^^^^^^^^^^^^^
All :ref:`test functions <modules/SaQCFunctions:Test Functions>` available in the
:doc:`Python API <../gettingstarted/TutorialAPI>` are also available in the configuration
system. The translation between API calls and the configuration syntax is straight forward
and best described by an example. Let's consider the definition
of :py:meth:`flagRange <saqc.SaQC.flagRange>`:

.. code-block:: python

   flagRange(field, min=-inf, max=inf, flag=255.)

The signature consists of the prevalent parameter ``field``, the specific parameters ``min``
and ``max`` as well as the :ref:`global parameter <documentation/GlobalKeywords:dfilter and flag keywords>`
``flag``. The translation of the given API call to ``flagRange``

.. code-block:: python

   qc.flagRange("x", 0, 100, flag=BAD)


into the configuration syntax look as follows:

.. code-block::

   varname ; test
   #-------;------------------------------------
   x       ; flagRange(min=0, max=100, flag=BAD)

We made the following changes: The value for ``field`` is given in the first column of the
configuration file, the actual function including all parameter as name-value pairs are given
in the second column.

All other test functions can be used in the same manner.


Regular Expressions in ``varname`` column
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some of the tests (e.g. checks for missing values, range tests or interpolation
functions) are very likely to be used on all or at least several variables of
a given dataset. As it becomes quite cumbersome to list all these
variables seperately, only to call the same functions with the same
parameters, SaQC supports regular expressions on variables. To mark a given
variable name as a regular expression, it needs to be quoted with ``'`` or ``"``.

.. code-block::

   varname    ; test
   #----------;------------------------------
   '.*'       ; shift(freq="15Min")
   '(x | y)'  ; flagMissing()
